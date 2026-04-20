# Interruption-Based Multi-Agent Writing Helper

This project is a desktop writing assistant prototype built with Tkinter and AutoGen/OpenAI. It streams text, lets the user interrupt when a sentence feels wrong, guesses possible reasons for that interruption, proposes local replacement options, and learns from the user's choice through a concise preference profile plus an interruption/revision log.

## What Is Already Implemented

### 1. Modular project structure
The project is no longer a single large script.

Current layout:

- `writing.py` - small entry point
- `writing_helper/main.py` - startup and API-key check
- `writing_helper/ui.py` - Tkinter interface
- `writing_helper/orchestrator.py` - workflow coordination
- `writing_helper/agents.py` - writer, reason interpreter, replacement generator, preference memory
- `writing_helper/models.py` - shared dataclasses and session state
- `writing_helper/text_utils.py` - sentence extraction and JSON parsing helpers
- `writing_helper/constants.py` - shared limits and regex constants

### 2. Popup desktop UI
Implemented with `tkinter`.

Current UI includes:

- task / purpose input
- broad style goal input
- live document display
- system log
- interruption context display
- replacement option list
- custom reason input for `Other`
- user preference profile display
- buttons for:
  - Start Streaming
  - Stop Streaming
  - Accept Current Text
  - Continue Generation
  - Export Session JSON
  - Apply Selected Replacement

### 3. Streaming writer
Implemented with AutoGen and `model_client_stream=True`.

Current behavior:

- streams text into the UI
- can be interrupted by the user
- uses:
  - task
  - style goal
  - accepted text
  - current live text
  - user preference profile
  - recent interruption / revision history

### 4. Sentence-aware interruption handling
Implemented.

When the user stops generation, the system:

- extracts the interrupted current sentence
- also captures the previous sentence if available
- records the replacement start point for local rewriting

This context is shown in the UI and passed into the follow-up agents.

### 5. Reason interpreter agent
Implemented.

The `ReasonInterpreterAgent`:

- is stateless across calls
- looks at the writing goal, current sentence, previous sentence, and preference profile
- proposes likely reasons for interruption
- returns up to 10 reasons, usually targeting about 5
- excludes `Other` from generated reasons

There is also a local fallback path if JSON parsing or model output fails.

### 6. Replacement generator agent
Implemented.

The `ReplacementAgent`:

- is stateless across calls
- generates one local replacement per proposed reason
- focuses on rewriting only the interrupted sentence or immediate local segment
- adds an `Other` option separately in the orchestrator

### 7. Preference memory
Implemented.

The `PreferenceMemoryAgent`:

- stores a concise user preference profile
- updates that profile from the reason tied to the replacement the user selected
- adds custom text directly to the profile when the user chooses `Other`

The current design intentionally keeps the memory in one concise profile instead of giving memory to the reason or replacement agents.

### 8. Interruption and revision log
Implemented.

For each interruption, the system stores:

- previous sentence
- interrupted sentence
- potential reasons considered
- selected reason
- selected revision
- custom reason if `Other` is used
- updated preference profile at that point

This log is also fed back into the streaming writer for future generations.

### 9. Exportable session state
Implemented.

The session export includes:

- current state snapshot
- current replacement options
- revision log

## What Is Still Missing

These are the main gaps in the current prototype.

### 1. Persistent memory across runs
Not implemented.

Right now:

- preference profile is session-only
- interruption log is session-only
- all learned behavior is lost when the app closes unless exported manually

Possible next step:

- save/load JSON session files automatically
- or use sqlite for a more durable profile/history store

### 2. Stronger document segmentation
Partially implemented, but still simple.

Right now:

- interruption handling is sentence-local
- replacement is applied by replacing the interrupted tail from a computed start point

Still missing:

- paragraph-aware replacement
- selection of exact edit span beyond current heuristic
- stronger handling for incomplete sentences, lists, or unusual punctuation

### 3. Better revision-option ranking
Not implemented.

Right now:

- one replacement is generated per reason
- the options are displayed in returned order

Still missing:

- scoring / ranking replacements
- hiding weak duplicates
- selecting top-k based on quality rather than raw generation order

### 4. Better validation of LLM JSON output
Partially implemented.

Right now:

- the system extracts a JSON object from raw model text
- falls back to heuristic logic when parsing fails

Still missing:

- stricter schema validation
- more robust handling of malformed model output
- retries for invalid JSON

### 5. Persistent user profile management
Not implemented.

Still missing:

- editing or deleting learned preferences
- pinning / prioritizing preferences
- merging similar preferences
- profile import/export separate from full session export

### 6. Evaluation and analytics
Not implemented.

Still missing:

- user satisfaction metrics
- counts of accepted vs replaced revisions
- time-to-accept measurements
- side-by-side evaluation workflow

### 7. Runtime-tested streaming polish
Partially implemented, not fully verified.

The current code compiles and imports, but the exact AutoGen streaming event behavior should still be tested live with your API key and real generation runs.

### 8. README-level setup helpers
Still missing a few project niceties:

- `requirements.txt`
- `.gitignore`
- optional `.env` loading
- automated tests

## Current Workflow

The current interaction loop is:

1. user enters task and style goal
2. streaming writer generates text
3. user interrupts when the current sentence feels wrong
4. system extracts current and previous sentence
5. reason interpreter guesses likely interruption reasons
6. replacement generator produces one local replacement per reason
7. user selects one replacement, or chooses `Other`
8. selected reason or custom reason is added to the preference profile
9. interruption record is stored in the revision log
10. future streaming uses both:
   - preference profile
   - interruption / revision history

## Current Design Decisions

The prototype currently uses:

- one shared `SessionState`
- stateless reason and replacement agents
- one concise memory profile
- one interruption/revision log
- Tkinter desktop UI
- AutoGen/OpenAI for streaming and agent calls
- local heuristic fallbacks when LLM JSON output is missing or malformed

## Requirements

Install required packages:

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]" openai
```

`tkinter` is also required. On many Python installs it is already included.

### Python Version

Recommended:

- Python 3.10 or newer

## API Key

Set your OpenAI API key before running.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

### Windows CMD

```cmd
set OPENAI_API_KEY=your_key_here
```

### macOS / Linux

```bash
export OPENAI_API_KEY="your_key_here"
```

## Run

```bash
python writing.py
```

## Short Summary

### Already have

- modular package structure
- Tkinter desktop app
- streaming writer
- interruption by current sentence
- previous-sentence context
- stateless reason interpreter
- stateless replacement generator
- user-selectable local replacement options
- concise preference profile
- interruption / revision log reused by future generation
- session JSON export

### Still missing

- persistent storage across runs
- stronger edit-span control
- replacement ranking / deduplication
- stricter JSON/schema validation
- analytics and evaluation
- automated tests
- optional environment/config convenience files
