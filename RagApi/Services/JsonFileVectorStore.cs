using System.Collections.Concurrent;
using System.Text.Json;

namespace RagApi.Services;

/// <summary>
/// A simple JSON file backed vector store. Thread-safe and durable across process restarts.
/// </summary>
public class JsonFileVectorStore : IVectorStore
{
    private readonly string _filePath;
    private readonly ConcurrentDictionary<string, VectorItem> _items = new();
    private readonly object _persistLock = new();

    public JsonFileVectorStore(IConfiguration cfg)
    {
        var dir = cfg["VectorStore:Directory"];
        if (string.IsNullOrWhiteSpace(dir))
        {
            // Default under app data folder
            dir = Path.Combine(AppContext.BaseDirectory, "data");
        }
        Directory.CreateDirectory(dir);
        _filePath = Path.Combine(dir, cfg["VectorStore:FileName"] ?? "vectors.json");
        LoadFromDisk();
    }

    public void Upsert(string id, string text, float[] vector)
    {
        _items[id] = new VectorItem(id, text, vector);
        Persist();
    }

    public IReadOnlyList<VectorItem> All() => _items.Values.ToList();

    public IEnumerable<VectorItem> Search(float[] query, int topK = 3)
    {
        return _items.Values
            .Select(i => new { Item = i, Score = CosineSimilarity(query, i.Vector) })
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => x.Item);
    }

    private void LoadFromDisk()
    {
        try
        {
            if (!File.Exists(_filePath)) return;
            using var fs = File.OpenRead(_filePath);
            var payload = JsonSerializer.Deserialize<List<SerializableItem>>(fs, new JsonSerializerOptions(JsonSerializerDefaults.Web));
            if (payload == null) return;
            foreach (var s in payload)
            {
                _items[s.Id] = new VectorItem(s.Id, s.Text ?? string.Empty, s.Vector ?? Array.Empty<float>());
            }
        }
        catch
        {
            // ignore malformed file
        }
    }

    private void Persist()
    {
        lock (_persistLock)
        {
            var list = _items.Values.Select(v => new SerializableItem
            {
                Id = v.Id,
                Text = v.Text,
                Vector = v.Vector
            }).ToList();

            var tmp = _filePath + ".tmp";
            var json = JsonSerializer.Serialize(list, new JsonSerializerOptions(JsonSerializerDefaults.Web)
            {
                WriteIndented = true
            });
            File.WriteAllText(tmp, json);
            File.Copy(tmp, _filePath, overwrite: true);
            File.Delete(tmp);
        }
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) return -1f;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        var denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom == 0 ? 0 : (float)(dot / denom);
    }

    private sealed class SerializableItem
    {
        public string Id { get; set; } = string.Empty;
        public string? Text { get; set; }
        public float[]? Vector { get; set; }
    }
}
