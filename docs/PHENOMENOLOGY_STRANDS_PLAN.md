# Phenomenology of Perception → Strands Agents SDK Improvement Plan

## Part 1: Improvement Opportunities

### 1. Horizon Structure → Context/Conversation Management

**Phenomenological concept:** Every perception is "a totality open to a horizon of an indefinite number of perspectival views." The field of presence includes retention (what just passed), primal impression (the now), and protention (what's anticipated). Time is "a network of intentionalities," not a line.

**Current Strands limitation:** The `SlidingWindowConversationManager` treats context as a flat, linear buffer — oldest messages get dropped. The `SummarizingConversationManager` compresses but loses the *horizonal structure* — the sense that past context is still "held in grasp" at varying depths rather than either fully present or fully gone.

**Improvement opportunity: Horizonal Context Manager**
- Instead of a binary present/absent model, implement a *layered retention* system where older context is progressively compressed but never fully lost — each layer becomes "the retention of a retention, and the layer of time between it and myself becomes thicker"
- Recent messages: full fidelity (primal impression)
- Medium-past: summarized but with key tool results and decisions preserved (retention)
- Deep past: compressed to essential patterns, preferences, and learned constraints (sedimented horizon)
- Future-oriented: maintain a *protentional* structure — what the agent anticipates needing next, pre-fetching relevant context based on the trajectory of the conversation

### 2. Body Schema → Tool Design and Tool Reuse

**Phenomenological concept:** The body schema is a "dynamic, pre-reflective organizational structure" — not a representation *of* the body but the lived system *through which* the body engages the world. Tools become incorporated into the body schema (the blind man's cane, the driver's car). Habit is "a motor acquisition of new signification through the reworking and renewal of the body schema."

**Current Strands limitation:** Tools in Strands are statically registered — the agent has a flat list of tools with schemas. There's no concept of tools becoming "incorporated" through use, no learning of tool affordances through experience, and no dynamic adaptation of how tools are presented to the model based on the agent's history of using them.

**Improvement opportunity: Tool Incorporation / Adaptive Tool Schema**
- Track tool usage patterns and success/failure rates as "sedimented habit"
- Dynamically reorder, emphasize, or compress tool descriptions based on the agent's experience — frequently used tools get shorter descriptions (they're "incorporated"), rarely used tools get fuller ones
- Implement a "system of equivalences" — when a tool fails, the agent should have pre-reflective knowledge of equivalent tools that can achieve the same goal (like the body schema's ability to transpose motor tasks across orientations)
- Tool "warm-up": when an agent first encounters a new tool, provide richer guidance; as it develops "habit," reduce scaffolding

### 3. Figure/Ground (Gestalt) → Tool Selection and LLM Interaction

**Phenomenological concept:** Perception is always structured as figure against ground — never isolated atoms. The indeterminate background is a "positive phenomenon," not a deficiency. The Gestalt concept of *Gestaltung* reveals "phenomena as prior to things."

**Current Strands limitation:** The agent presents ALL tools to the model in every call, treating them as a flat, undifferentiated list. There's no figure/ground structure — no way to foreground relevant tools while keeping others as background.

**Improvement opportunity: Gestalt Tool Presentation**
- Implement dynamic tool foregrounding: based on conversation context, present a small set of "figure" tools with full schemas, while keeping others as a compressed "ground" (just names/one-liners)
- The model can "pull" background tools into the foreground when needed (like the boat mast suddenly "locking" to the hull)
- This reduces token usage AND improves tool selection accuracy by reducing cognitive load on the model

### 4. Motor Intentionality / Intentional Arc → Event Loop and Agent Planning

**Phenomenological concept:** Motor intentionality is the body's directedness toward the world through movement, *prior to intellectual representation*. The intentional arc integrates "past, future, human milieu, physical situation." When it "goes limp" (Schneider's case), abstract movement fails while concrete habitual movement persists.

**Current Strands limitation:** The event loop is reactive — it processes one model response at a time, executes tools, and loops. There's no "motor project" — no pre-reflective anticipation of what sequence of actions will be needed. The agent can't distinguish between "concrete" (habitual, well-practiced) and "abstract" (novel, requiring projection into the possible) tasks.

**Improvement opportunity: Intentional Arc in the Event Loop**
- Add a lightweight planning/anticipation layer that maintains a "motor project" — a pre-reflective sense of where the conversation is heading
- For habitual patterns (e.g., "read file → edit → test" in coding), allow the agent to execute with less deliberation (like Schneider's concrete movement)
- For novel tasks, trigger deeper deliberation (abstract movement requiring projection into the possible)
- The intentional arc should integrate temporal context (what happened), situational context (current state), and goal context (where we're heading) into a unified structure that shapes each model call

### 5. Perspective / Situated Finitude → Multi-Agent Systems

**Phenomenological concept:** "All seeing is necessarily from somewhere, and a view from nowhere would not be a view at all." Perspective is "ubiquitous — not just in sense experience, but in our intellectual, social, personal, cultural, and historical self-understanding." The *géométral* (view from nowhere) is incoherent.

**Current Strands limitation:** In multi-agent systems (swarm, graph), agents share context through message passing but don't have explicit *perspectives*. There's no concept of each agent having a situated viewpoint that shapes what it attends to.

**Improvement opportunity: Perspectival Multi-Agent Architecture**
- Each agent in a multi-agent system should have an explicit "perspective" — a set of priorities, attention filters, and interpretive biases that shape how it processes shared information
- Instead of trying to build one omniscient agent (the *géométral*), embrace perspectival multiplicity — multiple situated agents whose partial views compose into richer understanding
- Implement "intercorporeity" between agents — direct resonance/adjustment rather than explicit message-passing for coordination

### 6. Temporality (Retention-Protention) → Session Management

**Phenomenological concept:** When recalling a distant past, "I reopen time, I place myself back at a moment when it still included a horizon of the future that is today closed off." Time is not a line but a network.

**Current Strands limitation:** Session management stores/restores flat message lists. There's no temporal structure — no sense of which past moments were pivotal, which decisions closed off futures, or which patterns recur.

**Improvement opportunity: Temporal Session Architecture**
- Store sessions with temporal metadata: decision points, branching moments, recurring patterns
- When restoring a session, reconstruct not just the messages but the *temporal field* — what was anticipated, what was decided, what was deferred
- Enable "reopening time" — the ability to revisit a past decision point with awareness of what has happened since

### 7. Sedimentation → Cross-Session Learning

**Phenomenological concept:** "Our habitual, embodied engagements with the world accumulate and form a layered ground of meaning (sedimentation) that precedes and conditions deliberate acts."

**Current Strands limitation:** Each agent invocation starts fresh (aside from session restore). There's no accumulation of learned patterns across sessions.

**Improvement opportunity: Sedimented Knowledge Layer**
- Maintain a persistent "sedimented" layer that accumulates across sessions: common tool patterns, user preferences, domain-specific knowledge
- This layer conditions the agent's behavior without requiring explicit retrieval — it's "pre-reflective" background knowledge
- Connects to the steering/skills plugin system but goes deeper — not just explicit rules but learned dispositions

---

## Part 2: Benchmarks for Ablation

| Benchmark | What it measures | Which improvements to ablate | Link |
|---|---|---|---|
| **τ-bench / τ²-bench** | Multi-turn tool-agent-user interaction in realistic domains (retail, airline, telecom) | Horizonal context (#1), Tool incorporation (#2), Intentional arc (#4) | https://github.com/sierra-research/tau2-bench |
| **BFCL v3** | Multi-turn, multi-step function calling accuracy | Gestalt tool presentation (#3), Tool incorporation (#2) | https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html |
| **SWE-ContextBench** | Context reuse across related coding tasks — measures experience transfer | Sedimentation (#7), Temporal sessions (#6), Horizonal context (#1) | https://arxiv.org/abs/2602.08316 |
| **SWE-Bench-CL** | Continual learning for coding agents — knowledge accumulation, catastrophic forgetting | Sedimentation (#7), Habit/tool reuse (#2) | https://arxiv.org/html/2507.00014 |
| **AgentBench** | Multi-turn open-ended reasoning across 8 environments | Intentional arc (#4), Perspectival multi-agent (#5) | https://www.evidentlyai.com/blog/ai-agent-benchmarks |
| **MCP-Atlas** | Real multi-tool use via MCP — composition of multiple tools | Gestalt tool presentation (#3), System of equivalences (#2) | https://scale.com/blog/open-sourcing-mcp-atlas |
| **Context-Bench** | Agentic context engineering — multi-hop context retrieval | Horizonal context (#1), Temporal sessions (#6) | https://www.sundeepteki.org/blog/context-bench-a-benchmark-for-evaluating-agentic-context-engineering |
| **AgencyBench** | 6 core agentic capabilities across 32 real-world scenarios in 1M-token contexts | All improvements, especially horizonal context (#1) and intentional arc (#4) | https://arxiv.org/abs/2601.11044 |
| **Scale Agentic Tool Use (Enterprise)** | Multi-tool composition with process supervision | Tool incorporation (#2), Gestalt presentation (#3) | https://scale.com/leaderboard/tool_use_enterprise |

### Recommended Ablation Strategy

1. **Baseline**: Run original Strands agent on τ-bench, BFCL v3, and SWE-ContextBench
2. **Ablation A (Horizonal Context)**: Replace SlidingWindowConversationManager with the layered retention system → measure on τ-bench and Context-Bench
3. **Ablation B (Gestalt Tools)**: Implement dynamic tool foregrounding → measure on BFCL v3 and MCP-Atlas
4. **Ablation C (Sedimentation)**: Add cross-session learning → measure on SWE-ContextBench and SWE-Bench-CL
5. **Ablation D (Intentional Arc)**: Add anticipatory planning to event loop → measure on τ-bench and AgentBench
6. **Combined**: Stack all improvements → measure across all benchmarks

The most impactful and immediately implementable improvements are likely **#1 (Horizonal Context)** and **#3 (Gestalt Tool Presentation)** — they map cleanly to existing Strands extension points (ConversationManager and tool registry) and have direct benchmark coverage.

---

## Part 3: Benchmark Models & Configuration

**AWS Profile:** `ngde-science` (account 992382830173, role BedrockFullAccess)
**Region:** us-west-2 (cross-region inference)

| Model | Model ID | Use |
|---|---|---|
| Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Fast iteration, cost-efficient baseline |
| Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | Mid-tier evaluation |
| Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | Top-tier evaluation |

All verified working through Strands SDK on 2026-04-24.

---

## Part 4: Baseline Results (2026-04-24)

### Strands-native Benchmarks (Benchmarks 2 & 3)

| Model | Tool Accuracy | Tool Precision | Tool Recall | Context Retention |
|---|---|---|---|---|
| Haiku 4.5 | 100% | 100% | 100% | 100% |
| Sonnet 4.6 | 100% | 100% | 100% | 100% |
| Opus 4.6 | 100% | 100% | 100% | 100% |

**Finding:** Current benchmark tasks are too easy — all 3 models achieve ceiling performance.
This means we need harder variants to see differentiation from phenomenology-inspired improvements:
- **Tool calling:** Need 20+ tools, ambiguous prompts, multi-hop chains, tool failure recovery
- **Context retention:** Need 50+ facts, interleaved with distractor tasks, delayed recall after tool-heavy sequences
- **τ-bench (airline/retail):** Already harder — run next as primary benchmark

### τ-bench Baseline (Benchmark 1)
Status: Environment ready, script at `/home/yhzhangh/.workspace/run_tau_baselines.sh`
Command: `cd /home/yhzhangh/.workspace/tau2-bench && AWS_PROFILE=ngde-science bash /home/yhzhangh/.workspace/run_tau_baselines.sh`

### Next Steps
1. Run τ-bench baselines (airline + retail, 3 models, 3 trials each)
2. Design harder tool calling benchmark (20+ tools, ambiguous prompts, failure recovery)
3. Design harder context retention benchmark (50+ facts, distractor tasks, delayed recall)
4. Implement Thrust 1 (Horizonal Context Manager) and Thrust 2 (Adaptive Tool Schema)
5. Re-run all benchmarks with improvements
