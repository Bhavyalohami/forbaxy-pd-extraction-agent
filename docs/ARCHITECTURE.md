# Architecture

```mermaid
flowchart TD
    Client[DMS-MS / API Client] --> API[FastAPI Routes]
    API --> Chat[ChatService]
    Chat --> Agent[PrescriptionAgent]
    Agent --> Tools[ToolRegistry]
    Agent --> Memory[SessionStore]
    Agent --> Learn[ReviewLearningStore]
    API --> Parser[DocumentParser Interface]
    Parser --> Mock[MockParser Now]
    Parser -. future .-> LlamaParse[LlamaParseParser]
    Agent -. future .-> RAG[RAG / Vector Store]
    Agent -. future .-> MCP[MCP Tools]
```

The production boundary is the `DocumentParser`, `SessionStore`, `ReviewLearningStore`, and
`ToolRegistry` interface set. Business logic depends on these abstractions, not vendor SDKs.

