using System.Text;
using System.Text.Json;

namespace RagApi.Services;

/// <summary>
/// Provides a minimal Retrieval-Augmented Generation (RAG) workflow that can
/// index input texts into a vector store and retrieve the most
/// relevant texts for a given question using vector similarity.
/// </summary>
/// <remarks>
/// This implementation now also calls OpenAI Chat Completions to synthesize an
/// answer from retrieved context.
/// </remarks>
public class RagService
{
    private readonly IEmbeddingService _embedding;
    private readonly IVectorStore _store;
    private readonly IConfiguration _cfg;

    /// <summary>
    /// Creates a new <see cref="RagService"/> instance.
    /// </summary>
    /// <param name="embedding">Embedding provider used to vectorize text.</param>
    /// <param name="store">Vector store used for indexing and search.</param>
    /// <param name="cfg">Application configuration for OpenAI settings.</param>
    public RagService(IEmbeddingService embedding, IVectorStore store, IConfiguration cfg)
    {
        _embedding = embedding;
        _store = store;
        _cfg = cfg;
    }

    /// <summary>
    /// Computes an embedding for the supplied text and upserts it into the store under the given id.
    /// </summary>
    /// <param name="id">Unique identifier to associate with the text.</param>
    /// <param name="text">The text content to index.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task Index(string id, string text, CancellationToken ct = default)
    {
        var vec = await _embedding.Embed(text, ct);
        _store.Upsert(id, text, vec);
    }

    /// <summary>
    /// Retrieves the most relevant texts for the question by embedding the question
    /// and performing a vector similarity search over the indexed items, then asks
    /// OpenAI Chat to synthesize an answer from the retrieved context.
    /// </summary>
    /// <param name="question">The user question.</param>
    /// <param name="topK">Maximum number of relevant texts to include in the context.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<string> Ask(string question, int topK = 1, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(question))
            return "Empty question. Please provide a specific question.";

        var total = _store.All().Count;
        if (total == 0) return "No data. Please add documents first via /api/rag/index.";

        topK = Math.Clamp(topK, 1, total);
        var qvec = await _embedding.Embed(question, ct);
        var hits = _store.Search(qvec, topK).ToList();
        if (hits.Count == 0) return "No data. Please add documents first via /api/rag/index.";

        // Output limits
        const int maxContextChars = 2000;
        var perDocLimit = Math.Max(200, maxContextChars / hits.Count);

        static string TrimTo(string s, int max)
        {
            if (string.IsNullOrWhiteSpace(s)) return string.Empty;
            var t = s.Trim();
            if (t.Length <= max) return t;
            return t[..max].ReplaceLineEndings(" ") + "…";
        }

        static string Snippet(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return string.Empty;
            var t = s.Trim();
            var len = Math.Min(160, t.Length);
            var cut = t[..len].ReplaceLineEndings(" ");
            return len < t.Length ? cut + "…" : cut;
        }

        var numbered = hits.Select((h, i) => new { Index = i + 1, h.Id, h.Text }).ToList();
        var citations = string.Join("\n", numbered.Select(x => $"- [{x.Index}] {x.Id}: {Snippet(x.Text)}"));
        var context = string.Join("\n---\n", numbered.Select(x => $"[{x.Index}] ({x.Id})\n{TrimTo(x.Text, perDocLimit)}"));

        var systemPrompt = "You are an assistant that must answer strictly in English. Use only the provided context. Cite sources as [1], [2], … when appropriate. If the answer is not in the context, state that explicitly.";
        var userPrompt = $@"Question: {question}

        Quotes:
        {citations}

        Context:
        {context}";

        string answer;
        try
        {
            answer = await GenerateWithOpenAIAsync(systemPrompt, userPrompt, ct);
        }
        catch (Exception ex)
        {
            // Fallback: return context if generation fails
            answer = $"Failed to generate an answer ({ex.GetType().Name}). Below is the relevant context:\n\n{context}";
        }

        return $"Answer: {answer} Sources: {citations}";
    }

    private async Task<string> GenerateWithOpenAIAsync(string systemPrompt, string userPrompt, CancellationToken ct)
    {
        var apiKey = _cfg["OpenAI:ApiKey"] ?? string.Empty;
        if (string.IsNullOrWhiteSpace(apiKey))
            throw new InvalidOperationException("OpenAI API key is not configured. Set OpenAI:ApiKey in appsettings or environment.");

        var baseUrl = (_cfg["OpenAI:BaseUrl"] ?? "https://api.openai.com/v1").TrimEnd('/');
        var model = _cfg["OpenAI:ChatModel"] ?? "gpt-4o-mini";
        var temperature = float.TryParse(_cfg["OpenAI:Temperature"], out var t) ? t : 0.2f;
        var maxTokens = int.TryParse(_cfg["OpenAI:MaxTokens"], out var mt) ? mt : 700;

        var url = $"{baseUrl}/chat/completions";
        using var req = new HttpRequestMessage(HttpMethod.Post, url);
        req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);

        var payload = new
        {
            model,
            temperature,
            max_tokens = maxTokens,
            messages = new object[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userPrompt }
            }
        };

        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        req.Content = new StringContent(json, Encoding.UTF8, "application/json");

        using var http = new HttpClient();
        using var res = await http.SendAsync(req, ct);
        res.EnsureSuccessStatusCode();

        using var stream = await res.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);

        var content = doc.RootElement
            .GetProperty("choices")[0]
            .GetProperty("message")
            .GetProperty("content")
            .GetString();

        return content ?? string.Empty;
    }
}
