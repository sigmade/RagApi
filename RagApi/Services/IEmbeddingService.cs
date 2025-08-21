namespace RagApi.Services;

/// <summary>
/// Defines a contract for generating numeric vector representations (embeddings)
/// from natural language text.
/// </summary>
public interface IEmbeddingService
{
    /// <summary>
    /// Generates an embedding vector for the specified text input.
    /// </summary>
    /// <param name="input">The plain text to embed.</param>
    /// <param name="cancellationToken">A token to observe while waiting for the operation to complete.</param>
    /// <returns>A dense vector representing the semantic meaning of the input text.</returns>
    /// <remarks>
    /// Implementations typically call an external service or model to compute the vector.
    /// The dimensionality and value range are implementation-specific.
    /// </remarks>
    Task<float[]> Embed(string input, CancellationToken cancellationToken = default);
}
