using Microsoft.AspNetCore.Mvc;
using RagApi.Services;

namespace RagApi.Controllers;

[Route("api/[controller]")]
[ApiController]
public class RagController : ControllerBase
{
    private readonly RagService _rag;

    public RagController(RagService rag)
    {
        _rag = rag;
    }

    [HttpPost("Index")]
    public async Task<IActionResult> Index([FromBody] IndexRequest request, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request?.Id) || string.IsNullOrWhiteSpace(request.Text))
            return BadRequest("Id and Text are required");

        await _rag.Index(request.Id, request.Text, ct);
        return Ok(new { request.Id });
    }

    [HttpGet("Ask")]
    public async Task<IActionResult> Ask([FromQuery] string question, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(question))
            return BadRequest("The 'question' parameter is required");

        var answer = await _rag.Ask(question, ct: ct);
        return Ok(new { answer });
    }

    public sealed class IndexRequest
    {
        public string Id { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
    }
}
