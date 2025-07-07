# OpenAI Image generator

AI tool to create images from prompt

## Features

- FastAPI-based REST server
- Model Context Protocol (MCP) tool support
- OpenAI image generation tool
- Modern Python tooling (uv, ruff, mypy)
- Type hints and comprehensive testing
- Docker support for containerization
- Kubernetes deployment configuration

## Requirements

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. Clone this repository
2. Install dependencies using uv:

```bash
uv sync
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
uv sync
```

## Usage

### Running Locally

1. Start the server:

```bash
uv run service
```

2. Access the API:
   - API documentation: http://localhost:5000/docs
   - MCP server: http://localhost:5000/mcp
   - Available tools: http://localhost:5000/tools
   - Invoke a tool: POST http://localhost:5000/tools/{tool_name}/invoke

### Running with Docker

1. Build and start the container:

```bash
docker-compose up -d
```

2. Access the API:
   - Available tools: http://localhost:5000/tools
   - Invoke a tool: POST http://localhost:5000/tools/{tool_name}/invoke

3. Stop the container:

```bash
docker-compose down
```

### Deploying to Kubernetes

1. Apply the Kubernetes resource:

```bash
kubectl apply -f kubernetes/mbxai_resource.yaml
```

2. Check the deployment status:

```bash
kubectl get mbxairesource -n mbxai-sandbox
```

3. Access the service through the Kubernetes ingress or service.

## Development

- Run tests: `pytest`
- Type checking: `mypy src`
- Linting: `ruff check .`

## Adding New Tools

The MCP server is designed to make it easy to add new tools. Here's a step-by-step guide:

### 1. Create a New Tool File

Create a new Python file in the `src/openai_image_generator/project/` directory. For example, to create a calculator tool:

```bash
touch src/openai_image_generator/project/calculator.py
```

### 2. Define Your Tool Function with MCP Decorator

Implement your tool function with proper type hints and the MCP decorator:

```python
from typing import Any
from mcp.server.fastmcp import FastMCP

# Create a FastMCP instance for this module in project/calculate.py
mcp = FastMCP("calculator")

@mcp.tool()
async def calculate(operation: str, a: float, b: float) -> dict[str, Any]:
    """Perform a calculation based on the operation."""
    result = 0
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else "Error: Division by zero"

    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }

# Import tool in app.py
from ..project.calculate import calculate

# Export the tool for registration in app.py in register_tools
await server.add_tool(calculate)
```

### 3. Register Your Tool

Import and register your tool in `src/openai_image_generator/api/app.py`:

```python
from openai_image_generator.project.calculator import calculator_tool

# Add this line after the existing tool registration
mcp.register_tool(calculator_tool)
```

### 4. Test Your Tool

Restart the server and test your tool:

```bash
curl -X POST "http://localhost:5000/tools/calculate/invoke" \
     -H "Content-Type: application/json" \
     -d '{"operation": "add", "a": 5, "b": 3}'
```

Expected response:

```json
{
  "operation": "add",
  "a": 5,
  "b": 3,
  "result": 8
}
```

### 5. Add Tests

Create a test file for your tool in the `tests/` directory:

```python
import pytest
from openai_image_generator.project.calculator import calculate

@pytest.mark.asyncio
async def test_calculate():
    result = await calculate("add", 5, 3)
    assert result["operation"] == "add"
    assert result["a"] == 5
    assert result["b"] == 3
    assert result["result"] == 8
```

### Best Practices

- Keep tool functions focused on a single responsibility
- Use descriptive names for tools and parameters
- Provide comprehensive descriptions for tools and parameters
- Include proper error handling in your tool functions
- Add tests for all your tools
- Document any external dependencies or API keys required

## License

MIT
