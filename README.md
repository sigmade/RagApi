RagApi — a simple RAG API on .NET 8

Summary
- For demonstration, the vector store uses a JSON file at data/vectors.json. The file is pre-populated with sample data.
- To enable calls to OpenAI, set your API key in RagApi/appsettings.json under OpenAI:ApiKey.
- In development, Swagger is available at /swagger.

Setup
- Open RagApi/appsettings.json and insert your key:
  - OpenAI:ApiKey: YOUR_API_KEY
  - Optionally adjust OpenAI:BaseUrl, OpenAI:EmbeddingModel, OpenAI:ChatModel.
- Vector store settings:
  - VectorStore:Directory: "data"
  - VectorStore:FileName: "vectors.json" (already pre-filled for the example)

Run
- Navigate to the RagApi directory and run:
  - dotnet run
- Open your browser: https://localhost:PORT/swagger

Main endpoints
- POST /api/rag/index — index text (Id, Text)
- GET  /api/rag/ask?question=... — ask a question. The answer is composed from the nearest contexts and OpenAI Chat
