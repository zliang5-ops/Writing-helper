# Interruption-Based Multi-Agent Writing Helper

This README describes what is already implemented in `interruption_writing_helper.py`, what still needs to be defined, and which packages are required.

## Overview

This project is a single-file prototype of an interruption-based AI writing helper with a popup desktop frontend.

The current structure follows the Project C proposal:

- a **streaming main writer**
- a **monitor agent** that reacts after interruption
- a **memory agent** that records dissatisfaction and updates restrictions
- a **writing style agent** that infers style preferences
- a **revision agent** that generates a revised continuation
- an **orchestrator** that controls the workflow
- a **Tkinter frontend** for user interaction

The design matches the proposal logic:

1. user enters task and style goal  
2. main writer streams text  
3. user stops when content becomes unsatisfactory  
4. monitor agent proposes likely dissatisfaction reasons  
5. user selects a reason or types a custom one  
6. memory/style agents update constraints  
7. revision agent generates a revised continuation  
8. user can continue, accept, or interrupt again

---

## What is already done

### 1. Popup frontend window
Implemented with `tkinter`.

Current UI includes:

- task / purpose input box
- style goal input box
- live document display
- system / memory / state log
- dissatisfaction option panel
- custom reason text box
- current restrictions panel
- current style notes panel
- buttons for:
  - Start Streaming
  - Stop Streaming
  - Accept Current Text
  - Continue Generation
  - Export Session JSON

When the `.py` file is run, a separate desktop window opens.

---

### 2. Main streaming generation
The main writer agent is set up with AutoGen and `model_client_stream=True`.

Implemented in:

- `StreamingWriterAgent`

Current behavior:

- builds a prompt from task, style goal, accepted text, restrictions, and style notes
- streams text incrementally into the frontend
- can be interrupted by the user with the Stop button

---

### 3. Agent roles are explicitly separated
The file already contains these components:

- `MonitorAgent`
- `MemoryAgent`
- `WritingStyleAgent`
- `StreamingWriterAgent`
- `RevisionAgent`
- `WritingOrchestrator`

The flow is explicit:

`User -> Orchestrator -> StreamingWriterAgent -> interruption -> MonitorAgent -> user feedback -> MemoryAgent + WritingStyleAgent -> RevisionAgent -> back to main generation`

So the skeleton for who points to who is already built.

---

### 4. User input and interaction loop
Implemented.

The code currently accepts:

- writing task
- broad style goal
- dissatisfaction choice from monitor options
- custom dissatisfaction reason if options do not fit

The monitor agent already generates a list of possible reasons such as:

- Too generic
- Too formal
- Too repetitive
- Too long
- Not natural enough
- Off-topic
- Need more academic tone
- Need more concise wording
- Other

---

### 5. Memory/state update scaffold
Implemented.

The `MemoryAgent` currently:

- records dissatisfaction reason
- converts that into simple restriction rules
- updates the running session state

The `WritingStyleAgent` currently:

- infers simple style notes from style goal and feedback
- stores those notes in session state

The system can also export the whole session as JSON.

---

### 6. Revision path after interruption
Implemented.

After the user submits feedback:

- the system updates restrictions
- the system updates style notes
- the revision agent generates a revised continuation
- the updated state is visible in the UI

---

## What is still left for you to define

This file is intentionally a scaffold. The architecture is done, but the exact research logic is still yours to specify.

### 1. Exact dissatisfaction interpretation protocol
Still open.

You said you want to decide:

- how interruption should be interpreted
- when interruption counts as dissatisfaction
- what monitor options should be shown
- whether some options depend on the sentence where the stop happened
- how custom reasons should be normalized into structured labels

Right now the monitor logic is heuristic and simple.

---

### 2. Exact restriction format
Still open.

You said you want to decide:

- how to add restrictions
- whether restrictions are plain sentences, tags, JSON objects, or prompt blocks
- whether restrictions should expire later
- whether restrictions should be global or local to one revision region

Right now restrictions are stored as simple strings like:

- "Avoid generic filler and broad vague claims."
- "Prefer shorter sentences and faster point delivery."

---

### 3. Exact inter-agent communication protocol
Still open.

You said you want to decide:

- how agents should pass information to each other
- the prompt/message format between agents
- whether agents share one document/state object or separate state
- whether some agents should be LLM-driven and others rule-based

Right now they share one Python `SessionState` object.

That is enough for a prototype, but not yet a formal protocol.

---

### 4. Whether monitor/style/memory should become full LLM agents
Still open.

Currently:

- `StreamingWriterAgent` uses AutoGen + model client
- `RevisionAgent` reuses the same writer in revision mode
- `MonitorAgent`, `MemoryAgent`, and `WritingStyleAgent` are local Python logic components

That means the architecture is multi-component, but not every component is yet a separate LLM-backed AutoGen agent.

If you want, these can later be upgraded so each one has its own prompt and model behavior.

---

### 5. Stronger streaming event handling for your exact package version
Possibly needs adjustment.

The current file is written against current AutoGen-style interfaces, but streaming event object shapes sometimes differ slightly by package version.

So after installation, you may need small edits in the section that reads streamed chunks from `run_stream(...)`.

The architecture is already correct; this is just version-specific polishing.

---

### 6. Better document control logic
Still open.

You may later want to decide:

- whether revision replaces only the interrupted local segment
- whether revision appends to the current text
- whether accepted text and live text should be split more formally
- whether the user can choose among multiple revision candidates instead of one

Right now the revision output is appended into the running live text.

---

### 7. Persistent memory across runs
Not implemented yet.

Current memory is session-only.

That means when the app closes, restrictions and revision history are lost unless exported manually.

If you later want persistent memory, you can add:

- local JSON storage
- sqlite
- redis
- diskcache
- vector database
- another user profile file

---

### 8. Evaluation module
Not implemented yet.

Your proposal mentions evaluation through:

- user satisfaction survey
- outside readers judging whether output is more human-like

The current code does not yet include:

- survey forms
- logging of evaluation metrics
- side-by-side blind comparison workflow
- automatic tracking of revision counts / time saved / accepted outputs

---

## Current design decisions in the prototype

These are the current defaults used only to make the scaffold run:

- one shared `SessionState` object
- simple restriction strings
- simple style notes
- heuristic monitor logic
- one streaming writer model
- one revision mode using the same writer model
- Tkinter desktop app frontend
- background async loop for model calls

You can replace any of these later.

---

## Required packages

Install these first:

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]" openai
```

`tkinter` is also used, but on many Python installations it is already included.

If `tkinter` is missing:

### Windows
Usually bundled with standard Python from python.org.

### Ubuntu / Debian
```bash
sudo apt-get install python3-tk
```

### Conda
```bash
conda install tk
```

---

## Python version

Recommended:

- Python 3.10 or newer

---

## Environment variable

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

---

## Run the app

```bash
python interruption_writing_helper.py
```

---

## Files

Current files:

- `interruption_writing_helper.py` — main prototype
- `README.md` — this file

---

## Suggested next steps

A reasonable next sequence would be:

1. run the current file and confirm the UI opens
2. verify streaming works with your installed AutoGen version
3. decide the exact dissatisfaction label schema
4. decide the exact restriction format
5. decide whether monitor/style/memory stay rule-based or become LLM agents
6. decide whether all agents share one state document or separate protocol messages
7. add persistent storage if needed
8. add evaluation logging

---

## Short summary

### Already built
- popup frontend
- user input
- streaming main writer
- interruption button
- monitor feedback options
- memory/style update scaffold
- revision generation path
- exportable session state
- explicit agent wiring

### Still to define
- exact protocol for interpretation
- exact restriction format
- exact agent message format
- local vs shared document/state design
- persistent memory
- evaluation module
- package-version-specific stream polishing
