import json
import pytest
from unittest.mock import MagicMock, patch

from llm_providers import (
    BaseLLMProvider,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockProvider,
    extract_json_from_response,
    get_llm_provider,
)
from planner import generate_validated_plan
from schemas import CleaningAction, CleaningPlan


def test_extract_json_from_response():
    # Plain JSON
    assert extract_json_from_response('{"a": 1}') == '{"a": 1}'

    # Markdown json block
    md_json = '```json\n{"summary": "clean", "actions": []}\n```'
    assert extract_json_from_response(md_json) == '{"summary": "clean", "actions": []}'

    # Generic markdown block
    md_generic = '```\n{"summary": "clean"}\n```'
    assert extract_json_from_response(md_generic) == '{"summary": "clean"}'

    # Leading/trailing whitespace
    assert extract_json_from_response('   {"x": 10}   \n') == '{"x": 10}'


def test_mock_provider():
    plan_dict = {
        "summary": "Mock plan summary",
        "actions": [
            {
                "action": "normalize_whitespace",
                "column": "name",
                "params": {},
                "reasoning": "Remove trailing space",
            }
        ],
    }
    mock = MockProvider(plan_dict=plan_dict)
    assert mock.provider_name == "mock"
    assert mock.model_name == "mock-planner"

    res = mock.generate_plan_json("test prompt")
    parsed = json.loads(res)
    assert parsed["summary"] == "Mock plan summary"
    assert len(mock.call_history) == 1
    assert mock.call_history[0]["prompt"] == "test prompt"

    # Test dynamic update
    mock.set_plan({"summary": "Updated", "actions": []})
    res2 = mock.generate_plan_json("second prompt")
    assert json.loads(res2)["summary"] == "Updated"


def test_get_llm_provider_factory():
    # Mock
    p_mock = get_llm_provider("mock")
    assert isinstance(p_mock, MockProvider)
    assert p_mock.provider_name == "mock"

    # Ollama (air-gapped local)
    p_ollama = get_llm_provider("ollama", model_name="llama3.1", base_url="http://127.0.0.1:11434")
    assert isinstance(p_ollama, OllamaProvider)
    assert p_ollama.provider_name == "ollama"
    assert p_ollama.model_name == "llama3.1"
    assert p_ollama.base_url == "http://127.0.0.1:11434"

    # Unsupported provider
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider("invalid_provider_xyz")


def test_openai_provider_mocked():
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"summary": "OpenAI plan", "actions": []}\n```'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            api_key="sk-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        assert provider.provider_name == "openai"
        assert provider.base_url == "https://api.groq.com/openai/v1"

        res = provider.generate_plan_json("Clean this dataset")
        assert res == '{"summary": "OpenAI plan", "actions": []}'
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_anthropic_provider_mocked():
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"summary": "Claude plan", "actions": []}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(
            model_name="claude-3-5-haiku-20241022",
            api_key="sk-ant-test-key",
        )
        assert provider.provider_name == "anthropic"

        res = provider.generate_plan_json("Clean this profile")
        assert res == '{"summary": "Claude plan", "actions": []}'
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-haiku-20241022"


def test_ollama_provider_mocked():
    provider = OllamaProvider(model_name="mistral", base_url="http://localhost:11434")
    assert provider.provider_name == "ollama"

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_res = MagicMock()
        mock_res.json.return_value = {
            "message": {"content": '{"summary": "Local Ollama plan", "actions": []}'}
        }
        mock_res.raise_for_status.return_value = None
        mock_client.post.return_value = mock_res

        res = provider.generate_plan_json("Clean dataset offline")
        assert res == '{"summary": "Local Ollama plan", "actions": []}'

        mock_client.post.assert_called_once()
        post_url, post_kwargs = mock_client.post.call_args
        assert post_url[0] == "http://localhost:11434/api/chat"
        assert post_kwargs["json"]["format"] == "json"
        assert post_kwargs["json"]["model"] == "mistral"


def test_generate_validated_plan_with_provider():
    plan_dict = {
        "summary": "Clean plan with provider",
        "actions": [
            {
                "action": "normalize_whitespace",
                "column": "name",
                "params": {},
                "reasoning": "Clean whitespace",
            }
        ],
    }
    mock_prov = MockProvider(plan_dict=plan_dict)

    plan, raw_text = generate_validated_plan(
        provider=mock_prov,
        initial_prompt="Analyze profile",
        token_to_column={"name": "name"},
        max_retries=2,
    )
    assert plan.summary == "Clean plan with provider"
    assert len(plan.actions) == 1
    assert plan.actions[0].action == "normalize_whitespace"
    assert len(mock_prov.call_history) == 1
