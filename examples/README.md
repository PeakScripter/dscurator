# Example Templates

These YAML templates demonstrate various ways to customize DS Curator for different domains.

## Usage

```bash
# Use a template directly
python -m dscurator run --template examples/tourism_qa.yaml --model "gpt-4o-mini" --limit 50

# Or export and customize
python -m dscurator init-template --type qa --out my_template.yaml
# Edit my_template.yaml, then:
python -m dscurator run --template my_template.yaml --model "claude-3-5-sonnet"
```

## Available Examples

1. **tourism_qa.yaml** - Q&A pairs for tourism domain
2. **medical_cot.yaml** - Chain of thought for medical reasoning
3. **tech_agent.yaml** - Agent trajectory simulation for tech tasks

## Template Structure

```yaml
type: qa  # or corpus|cot|agent
seed: "your domain"
prompt_template: "Template with {topic} and {path} placeholders"
system: "System prompt for the LLM"

algorithm: bfs  # bfs|walk|clustering|markov|graph
filter:
  min_length: 50
  enforce_diversity: true
```

## Supported Fields

- **type**: Dataset type (corpus, qa, cot, agent)
- **seed**: Starting domain/topic
- **prompt_template**: Python f-string format with {topic} and {path}
- **system**: System message for LLM context
- **algorithm**: Prompt expansion algorithm
- **filter**: Quality filtering options

