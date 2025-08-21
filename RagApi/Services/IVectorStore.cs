namespace RagApi.Services;

/// <summary>
/// A simple vector store abstraction supporting upsert, listing and vector similarity search.
/// </summary>
public interface IVectorStore
{
    /// <summary>
    /// Inserts a new item or updates an existing one using the provided id.
    /// </summary>
    void Upsert(string id, string text, float[] vector);

    /// <summary>
    /// Returns a snapshot of all items currently stored.
    /// </summary>
    IReadOnlyList<VectorItem> All();

    /// <summary>
    /// Searches the store for items with the highest cosine similarity to the query vector.
    /// </summary>
    IEnumerable<VectorItem> Search(float[] query, int topK = 3);
}

/// <summary>
/// Represents a stored item consisting of an identifier, the original text,
/// and its associated embedding vector.
/// </summary>
/// <param name="Id">Application-level identifier of the item.</param>
/// <param name="Text">Original text payload.</param>
/// <param name="Vector">Embedding vector for similarity search.</param>
public readonly record struct VectorItem(string Id, string Text, float[] Vector);
