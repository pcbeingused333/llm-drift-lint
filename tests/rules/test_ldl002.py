import pytest

from llm_drift_lint.rules.ldl002_sampling_kwargs_sdk1 import AnthropicSamplingKwargsSdk1 as Rule

# reward-bench#290, rewardbench/generative.py before the fix.
JUDGE = """
import anthropic

def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    c = anthropic.Anthropic()
    response = c.messages.create(
        model=model,
        messages=conv.messages,
        max_tokens=max_tokens,
        temperature=temperature,
        system=sys_msg,
    )
    return response
"""


def project(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    return pkg / "judge.py"


@pytest.mark.parametrize(
    "files",
    [
        {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["anthropic>=1.0"]\n'},
        {
            "pyproject.toml": '[project]\nname = "x"\n[project.optional-dependencies]\napi = ["anthropic~=1.13"]\n'
        },
        {"requirements.txt": "anthropic==1.13.0\n"},
        # reward-bench: unpinned in pyproject, 1.x in the lockfile.
        {
            "pyproject.toml": '[project]\nname = "x"\ndependencies = ["anthropic"]\n',
            "uv.lock": 'version = 1\n[[package]]\nname = "anthropic"\nversion = "1.13.0"\n',
        },
    ],
)
def test_sampling_kwarg_with_anthropic_1x(lint, tmp_path, files):
    findings = lint(Rule, JUDGE, project(tmp_path, files))
    assert [(f.rule_id, f.line) for f in findings] == [("LDL002", 6)]
    assert "temperature" in findings[0].message


def test_every_sampling_kwarg_and_request_method(lint, tmp_path):
    path = project(tmp_path, {"requirements.txt": "anthropic>=1,<2\n"})
    findings = lint(
        Rule,
        """
        from anthropic import AsyncAnthropic
        async def go(client):
            await client.messages.create(model="m", max_tokens=1, top_p=0.9, top_k=5)
            async with client.messages.stream(model="m", max_tokens=1, temperature=None) as s:
                pass
        """,
        path,
    )
    assert [(f.line, f.message.split("`")[1]) for f in findings] == [
        (4, "top_p"),
        (4, "top_k"),
        (5, "temperature"),
    ]


@pytest.mark.parametrize(
    "files",
    [
        {},  # no project files at all
        {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["anthropic"]\n'},  # unbounded
        {"requirements.txt": "anthropic>=0.40\n"},
        {"requirements.txt": "anthropic<1\n"},
        {
            "pyproject.toml": '[project]\nname = "x"\ndependencies = ["anthropic>=0.40"]\n',
            "uv.lock": 'version = 1\n[[package]]\nname = "anthropic"\nversion = "0.69.0"\n',
        },
        {"requirements.txt": "langchain-anthropic>=1.0\n"},  # a different package
    ],
)
def test_silent_unless_the_project_requires_anthropic_1x(lint, tmp_path, files):
    assert not lint(Rule, JUDGE, project(tmp_path, files))


def test_silent_without_a_real_path(lint):
    assert not lint(Rule, JUDGE)


def test_extra_body_and_other_files_are_silent(lint, tmp_path):
    path = project(tmp_path, {"requirements.txt": "anthropic>=1\n"})
    # The fixed shape: sampling parameters in the request body.
    assert not lint(
        Rule,
        """
        import anthropic
        c.messages.create(model=model, max_tokens=8, extra_body={"temperature": temperature})
        """,
        path,
    )
    # A file that doesn't use the anthropic SDK (e.g. a wrapper with its own messages.create).
    assert not lint(Rule, "client.messages.create(model=model, temperature=0.0)\n", path)
    # Other APIs.
    assert not lint(
        Rule,
        """
        import anthropic
        openai_client.chat.completions.create(model=model, temperature=0.0)
        """,
        path,
    )
