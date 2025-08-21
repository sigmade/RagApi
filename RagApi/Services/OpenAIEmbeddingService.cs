using System.Text;
using System.Text.Json;

namespace RagApi.Services;

/// <summary>
/// Embedding service implementation that calls the OpenAI Embeddings REST API
/// to generate semantic vector representations for input text.
/// </summary>
/// <remarks>
/// This service sends HTTP POST requests to the configured OpenAI endpoint and parses
/// the returned JSON to extract the embedding vector. The default model used is
/// <c>text-embedding-3-small</c>, but it can be overridden via configuration.
/// Expected configuration keys:
/// <list type="bullet">
/// <item><description><c>OpenAI:ApiKey</c> - the API key used for bearer authentication.</description></item>
/// <item><description><c>OpenAI:BaseUrl</c> - the base URL of the API (defaults to https://api.openai.com/v1).</description></item>
/// <item><description><c>OpenAI:EmbeddingModel</c> - the model id to use for embeddings.</description></item>
/// </list>
/// </remarks>
public class OpenAIEmbeddingService : IEmbeddingService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly string _baseUrl;
    private readonly string _model;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    /// <summary>
    /// Initializes a new instance of the <see cref="OpenAIEmbeddingService"/>.
    /// </summary>
    /// <param name="configuration">Application configuration used to resolve OpenAI settings.</param>
    public OpenAIEmbeddingService(IConfiguration configuration)
    {
        _httpClient = new HttpClient();
        _apiKey = configuration["OpenAI:ApiKey"] ?? string.Empty;
        _baseUrl = configuration["OpenAI:BaseUrl"] ?? "https://api.openai.com/v1";
        _model = configuration["OpenAI:EmbeddingModel"] ?? "text-embedding-3-small";
    }

    /// <summary>
    /// Requests an embedding vector for the supplied input text from the OpenAI API.
    /// </summary>
    /// <param name="input">The text to embed.</param>
    /// <param name="cancellationToken">A token to cancel the request.</param>
    /// <returns>A floating-point vector representing the semantic embedding of the input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the OpenAI API key is not configured.</exception>
    /// <exception cref="HttpRequestException">Thrown when the HTTP call fails or the response indicates an error.</exception>
    /// <remarks>
    /// For more details about the API contract, refer to the OpenAI Embeddings endpoint documentation.
    /// </remarks>
    public async Task<float[]> Embed(string input, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(_apiKey))
        {
            throw new InvalidOperationException("OpenAI API key is not configured. Set OpenAI:ApiKey in appsettings or environment.");
        }

        var url = $"{_baseUrl.TrimEnd('/')}/embeddings";
        using var req = new HttpRequestMessage(HttpMethod.Post, url);
        req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);

        var payload = new
        {
            input,
            model = _model
        };
        req.Content = new StringContent(JsonSerializer.Serialize(payload, JsonOptions), Encoding.UTF8, "application/json");

        using var res = await _httpClient.SendAsync(req, cancellationToken);
        res.EnsureSuccessStatusCode();

        using var stream = await res.Content.ReadAsStreamAsync(cancellationToken);
        var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var first = doc.RootElement.GetProperty("data")[0].GetProperty("embedding");
        var arr = new float[first.GetArrayLength()];
        var i = 0;
        foreach (var v in first.EnumerateArray())
        {
            arr[i++] = v.GetSingle();
        }
        return arr;
    }
}
