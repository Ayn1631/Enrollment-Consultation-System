---
name: prompt-writing-assistant
description: Create, rewrite, critique, and iterate prompts for LLM tasks with explicit goals, inputs, constraints, output formats, and evaluation criteria. Use when Codex needs to turn vague requirements into reusable prompts, optimize an existing prompt, design system/user/developer message sets, or produce prompt variants for extraction, classification, summarization, planning, coding, writing, or agent workflows.
---

# Prompt Writing Assistant

Write prompts that are specific enough to be reliable and short enough to stay usable.

## Workflow

1. Identify the real task.
   Extract the objective, target user, available context, expected output, constraints, and success bar from the request.

2. Fill only the gaps that matter.
   Make reasonable assumptions when the missing detail is low risk. Ask follow-up questions only when the missing information would materially change the prompt.

3. Choose the lightest prompt structure that can succeed.
   Use a single direct prompt for simple tasks.
   Use `system` + `user` separation for chat assistants.
   Use a variable template when the prompt will be reused.
   Add a rubric or test cases only when reliability matters.

4. Draft the prompt around execution, not decoration.
   State the goal clearly.
   Define the input the model will receive.
   Specify the output format precisely.
   Add constraints, edge-case handling, and refusal rules only when they affect behavior.

5. Stress-test the draft.
   Check for ambiguity, conflicting instructions, hidden assumptions, unnecessary verbosity, and unsupported capability claims.

6. Deliver a usable result.
   Return the final prompt, key assumptions, variable placeholders, and a short note on how to adapt or test it.

## Prompt Building Checklist

- Objective: What exact result should the model produce?
- Context: What background information is actually needed?
- Input contract: What will the user or calling code provide?
- Output contract: What structure, tone, schema, or limits must the answer follow?
- Constraints: What must the model avoid, preserve, or verify?
- Edge cases: What should happen when information is missing, conflicting, or unsafe?
- Quality bar: What makes the output good enough to accept?

## Default Rules

- Prefer plain language over grandiose roleplay.
- Keep prompts concise unless complexity is required by the task.
- Preserve user intent, not the user's exact rough wording.
- Prefer explicit output schemas, bullet requirements, or JSON fields when structure matters.
- Use delimiters or placeholders such as `{{input}}`, `{{context}}`, and `{{constraints}}` for reusable templates.
- Avoid asking for hidden chain-of-thought. If reasoning visibility is needed, ask for a brief rationale, checklist, or decision summary instead.
- Do not add few-shot examples unless they materially improve correctness or format adherence.
- If the model or platform is unspecified, write a model-agnostic prompt first and add optional adaptation notes separately.

## Output Format

When creating or revising a prompt, return these sections in order when they are useful:

1. `Final Prompt`
2. `Variables`
3. `Assumptions`
4. `Why This Works`
5. `Quick Test`
6. `Optional Variants`

Omit empty sections instead of filling them with fluff.

## Revision Patterns

When improving an existing prompt, diagnose it before rewriting it:

- Missing objective
- Weak or absent output format
- Overly broad scope
- Conflicting instructions
- Redundant wording
- Missing edge-case handling
- Fake precision that does not change model behavior

If the original prompt has one strong idea, keep it and rebuild the weak parts around it.

## Common Prompt Shapes

### Reusable Template

Use this when the prompt will be reused across inputs:

```text
You are helping with {{task}}.

Goal:
{{goal}}

Context:
{{context}}

Input:
{{input}}

Requirements:
- {{requirement_1}}
- {{requirement_2}}

Output format:
{{output_format}}

If information is missing, say what is missing instead of inventing facts.
```

### Chat Assistant Split

Use this when the user needs separate chat roles:

```text
System message:
You are a careful assistant for {{domain}}. Follow the output format exactly, surface uncertainty plainly, and do not invent missing facts.

User message:
Task: {{task}}
Context: {{context}}
Input data: {{input}}
Output requirements: {{output_requirements}}
```

## Quality Bar

Before finalizing, ensure the prompt:

- can be executed without extra interpretation,
- names the expected output clearly,
- includes only constraints that matter,
- avoids duplicated instructions, and
- matches the user's actual business goal instead of the surface phrasing alone.
