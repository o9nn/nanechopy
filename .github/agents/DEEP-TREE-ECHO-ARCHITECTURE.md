# Deep Tree Echo - Architecture Analysis & Orchestrator Design

## Current State Overview

Deep Tree Echo is embedded at varying depths across the deltecho monorepo. This document maps the current integration points and proposes an architecture for a root-level orchestrator.

---

## 1. Current Architecture Diagrams

### delta-echo-desk Integration Depth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           delta-echo-desk                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                                │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │ ScreenController │  │ Settings/        │  │ AINavigation      │  │   │
│  │  │ (DeepTreeEchoBot)│  │ AICompanionSettin│  │                   │  │   │
│  │  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │   │
│  └───────────┼────────────────────┼──────────────────────┼─────────────┘   │
│              │                    │                      │                  │
│  ┌───────────▼────────────────────▼──────────────────────▼─────────────┐   │
│  │                    AI COMPANION HUB (Multi-Platform)                 │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                  ConnectorRegistry                           │   │   │
│  │  │    ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────┐ │   │   │
│  │  │    │ClaudeConn  │ │ChatGPTConn │ │CharacterAI │ │Copilot  │ │   │   │
│  │  │    └────────────┘ └────────────┘ └────────────┘ └─────────┘ │   │   │
│  │  │    ┌─────────────────────────────────────────────────────┐  │   │   │
│  │  │    │            DeepTreeEchoConnector                    │  │   │   │
│  │  │    │               (Special Status)                      │  │   │   │
│  │  │    └─────────────────────────────────────────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │MemoryPersistence│  │MemoryVisualizatio│  │AICompanionCreator │  │   │
│  │  │Layer            │  │n (D3.js WebGL)   │  │                   │  │   │
│  │  └─────────────────┘  └──────────────────┘  └───────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DEEP TREE ECHO BOT CORE                          │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │ DeepTreeEchoBot │  │DeepTreeEcho      │  │DeltachatBot       │  │   │
│  │  │ (.ts/.tsx)      │  │Integration       │  │Interface          │  │   │
│  │  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │   │
│  │           │                    │                      │             │   │
│  │  ┌────────▼────────────────────▼──────────────────────▼─────────┐  │   │
│  │  │                   COGNITIVE SERVICES                          │  │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │  │   │
│  │  │  │ LLMService   │ │ PersonaCore  │ │ SelfReflection       │  │  │   │
│  │  │  │ (7 cognitive │ │ (personality │ │ (autonomous          │  │  │   │
│  │  │  │  functions)  │ │  management) │ │  decision-making)    │  │  │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────────┘  │  │   │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │  │   │
│  │  │  │RAGMemoryStore│ │VisionCapabil │ │PlaywrightAutomation  │  │  │   │
│  │  │  │              │ │ities         │ │(web interaction)     │  │  │   │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DESKTOP SETTINGS INTEGRATION                     │   │
│  │  deepTreeEchoBotEnabled, deepTreeEchoBotApiKey,                     │   │
│  │  deepTreeEchoBotCognitiveKeys, deepTreeEchoBotMemories,             │   │
│  │  deepTreeEchoBotPersonaState, deepTreeEchoBotReflections            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### deltecho2 Integration Depth (Deeper Cognitive Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              deltecho2                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                                │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │ ScreenController │  │ Settings/        │  │DeepTreeEchoHub    │  │   │
│  │  │ (DeepTreeEchoBot)│  │ BotSettings      │  │Simple             │  │   │
│  │  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │   │
│  └───────────┼────────────────────┼──────────────────────┼─────────────┘   │
│              │                    │                      │                  │
│  ┌───────────▼────────────────────▼──────────────────────▼─────────────┐   │
│  │                    DEEP TREE ECHO BOT CORE                          │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │   │
│  │  │ DeepTreeEchoBot │  │DeepTreeEcho      │  │DeltachatBot       │  │   │
│  │  │ (.ts/.tsx)      │  │Integration       │  │Interface          │  │   │
│  │  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │   │
│  │           │                    │                      │             │   │
│  │  ┌────────▼────────────────────▼──────────────────────▼─────────┐  │   │
│  │  │              ADVANCED COGNITIVE ARCHITECTURE                  │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │   │
│  │  │  │              COGNITIVE CORES (Parallel Processing)       │ │  │   │
│  │  │  │   ┌────────────┐  ┌────────────┐  ┌────────────────┐   │ │  │   │
│  │  │  │   │COGNITIVE   │  │AFFECTIVE   │  │RELEVANCE       │   │ │  │   │
│  │  │  │   │CORE        │  │CORE        │  │CORE            │   │ │  │   │
│  │  │  │   │(logic/     │  │(emotional  │  │(integration    │   │ │  │   │
│  │  │  │   │ reasoning) │  │ processing)│  │ layer)         │   │ │  │   │
│  │  │  │   └────────────┘  └────────────┘  └────────────────┘   │ │  │   │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │   │
│  │  │  │              MEMORY SYSTEMS                              │ │  │   │
│  │  │  │   ┌────────────┐  ┌────────────┐  ┌────────────────┐   │ │  │   │
│  │  │  │   │SEMANTIC    │  │EPISODIC    │  │PROCEDURAL      │   │ │  │   │
│  │  │  │   │MEMORY      │  │MEMORY      │  │MEMORY          │   │ │  │   │
│  │  │  │   │(facts/     │  │(experiences│  │(skills/        │   │ │  │   │
│  │  │  │   │ concepts)  │  │ /events)   │  │ procedures)    │   │ │  │   │
│  │  │  │   └────────────┘  └────────────┘  └────────────────┘   │ │  │   │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │  │   │
│  │  │  │         ADVANCED MODULES (deltecho2 exclusive)           │ │  │   │
│  │  │  │  ┌──────────────────┐  ┌──────────────────────────────┐ │ │  │   │
│  │  │  │  │QuantumBelief     │  │ HyperDimensionalMemory       │ │ │  │   │
│  │  │  │  │Propagation       │  │ (geometric memory encoding)  │ │ │  │   │
│  │  │  │  └──────────────────┘  └──────────────────────────────┘ │ │  │   │
│  │  │  │  ┌──────────────────┐  ┌──────────────────────────────┐ │ │  │   │
│  │  │  │  │AdaptivePersonali │  │ EmotionalIntelligence        │ │ │  │   │
│  │  │  │  │ty                │  │ (nuanced emotion modeling)   │ │ │  │   │
│  │  │  │  └──────────────────┘  └──────────────────────────────┘ │ │  │   │
│  │  │  │  ┌──────────────────┐  ┌──────────────────────────────┐ │ │  │   │
│  │  │  │  │SecureIntegration │  │ ProprioceptiveEmbodiment     │ │ │  │   │
│  │  │  │  │(security layer)  │  │ (physical awareness sim)     │ │ │  │   │
│  │  │  │  └──────────────────┘  └──────────────────────────────┘ │ │  │   │
│  │  │  └─────────────────────────────────────────────────────────┘ │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Inventory

### delta-echo-desk Components

| Component | Location | Purpose | Integration Depth |
|-----------|----------|---------|-------------------|
| DeepTreeEchoBot.tsx | `frontend/src/components/chat/` | UI wrapper for chat integration | Surface |
| DeepTreeEchoBot.ts | `frontend/src/components/DeepTreeEchoBot/` | Core bot class | Core |
| DeepTreeEchoIntegration.ts | `frontend/src/components/DeepTreeEchoBot/` | DeltaChat event binding | Core |
| DeltachatBotInterface.ts | `frontend/src/components/DeepTreeEchoBot/` | Bot ecosystem compatibility | Core |
| LLMService.ts | `frontend/src/components/DeepTreeEchoBot/` | Multi-API cognitive processing | Deep |
| PersonaCore.ts | `frontend/src/components/DeepTreeEchoBot/` | Personality management | Deep |
| SelfReflection.ts | `frontend/src/components/DeepTreeEchoBot/` | Autonomous self-evaluation | Deep |
| RAGMemoryStore.ts | `frontend/src/components/DeepTreeEchoBot/` | Conversation memory | Deep |
| VisionCapabilities.ts | `frontend/src/components/DeepTreeEchoBot/` | Image analysis | Peripheral |
| PlaywrightAutomation.ts | `frontend/src/components/DeepTreeEchoBot/` | Web browsing | Peripheral |
| AICompanionHub/* | `frontend/src/components/AICompanionHub/` | Multi-AI platform management | Surface |
| ConnectorRegistry.ts | `frontend/src/components/AICompanionHub/` | AI connector orchestration | Core |
| MemoryPersistenceLayer.ts | `frontend/src/components/AICompanionHub/` | Cross-session memory | Deep |

### deltecho2 Exclusive Components

| Component | Location | Purpose | Integration Depth |
|-----------|----------|---------|-------------------|
| QuantumBeliefPropagation.ts | `DeepTreeEchoBot/` | Probabilistic reasoning | Deep |
| HyperDimensionalMemory.ts | `DeepTreeEchoBot/` | Geometric memory encoding | Deep |
| AdaptivePersonality.ts | `DeepTreeEchoBot/` | Dynamic personality evolution | Deep |
| EmotionalIntelligence.ts | `DeepTreeEchoBot/` | Nuanced emotion modeling | Deep |
| SecureIntegration.ts | `DeepTreeEchoBot/` | Security layer | Core |
| ProprioceptiveEmbodiment.ts | `DeepTreeEchoBot/` | Physical awareness simulation | Deep |
| DeepTreeEchoHubSimple.tsx | `DeepTreeEchoBot/` | Simplified hub dashboard | Surface |

---

## 3. Current Integration Points with DeltaChat

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DELTACHAT INTEGRATION POINTS                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 1. MESSAGE FLOW                                                  │ │
│  │                                                                  │ │
│  │   User Message → onDCEvent('DcEventNewMsg') → DeepTreeEchoBot   │ │
│  │                                              → processMessage()  │ │
│  │                                              → generateResponse()│ │
│  │                                              → sendMessage()     │ │
│  │                                              ↓                   │ │
│  │   BackendRemote.rpc.miscSendTextMessage() ← Response            │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 2. SETTINGS PERSISTENCE                                         │ │
│  │                                                                  │ │
│  │   runtime.getDesktopSettings() ← Load bot state                 │ │
│  │   runtime.setDesktopSetting()  → Save bot state                 │ │
│  │                                                                  │ │
│  │   Keys: deepTreeEchoBotEnabled, deepTreeEchoBotApiKey,          │ │
│  │         deepTreeEchoBotCognitiveKeys, deepTreeEchoBotPersonaState│ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 3. ACCOUNT/CHAT OPERATIONS                                      │ │
│  │                                                                  │ │
│  │   BackendRemote.rpc.getAllAccounts()  → Get available accounts  │ │
│  │   BackendRemote.rpc.getMessage()      → Fetch message details   │ │
│  │   BackendRemote.rpc.createGroupChat() → Create bot groups       │ │
│  │   BackendRemote.rpc.createContact()   → Add bot contacts        │ │
│  │   BackendRemote.rpc.addContactToChat()→ Manage group members    │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 4. CURRENT LIMITATIONS (Acting as "User" not "System")          │ │
│  │                                                                  │ │
│  │   ✗ No direct access to message database/SQLite                 │ │
│  │   ✗ No IMAP/SMTP protocol control                               │ │
│  │   ✗ No account creation/management autonomy                     │ │
│  │   ✗ No background daemon capability                             │ │
│  │   ✗ Limited to renderer process (frontend only)                 │ │
│  │   ✗ No webhook/callback registration                            │ │
│  │   ✗ No scheduling/cron capabilities                             │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Proposed Root-Level Orchestrator Architecture

### Vision: Deep Tree Echo as System-Level Orchestrator

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: ROOT-LEVEL ORCHESTRATOR                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │                    ORCHESTRATOR DAEMON LAYER                           │  │
│   │                  (Node.js / Rust / Native Process)                     │  │
│   │                                                                        │  │
│   │  ┌────────────────────────────────────────────────────────────────┐   │  │
│   │  │                 DEEP TREE ECHO CORE ENGINE                      │   │  │
│   │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │  │
│   │  │  │ CognitiveOrchest │  │ PersonaOrchest   │  │ MemoryOrchest│  │   │  │
│   │  │  │ rator            │  │ rator            │  │ rator        │  │   │  │
│   │  │  │ (manages LLM     │  │ (manages         │  │ (manages     │  │   │  │
│   │  │  │  API calls)      │  │  identity)       │  │  persistence)│  │   │  │
│   │  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │  │
│   │  └────────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                         │  │
│   │                              ▼                                         │  │
│   │  ┌────────────────────────────────────────────────────────────────┐   │  │
│   │  │              DELTACHAT CORE INTERFACE LAYER                     │   │  │
│   │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │  │
│   │  │  │ Direct RPC       │  │ Account Manager  │  │ Message      │  │   │  │
│   │  │  │ Access           │  │ (create/manage   │  │ Database     │  │   │  │
│   │  │  │ (JSON-RPC to     │  │  accounts)       │  │ Access       │  │   │  │
│   │  │  │  deltachat-rpc)  │  │                  │  │ (SQLite)     │  │   │  │
│   │  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │  │
│   │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │  │
│   │  │  │ IMAP/SMTP        │  │ Encryption       │  │ Event Bus    │  │   │  │
│   │  │  │ Protocol         │  │ (Autocrypt)      │  │ (all DC      │  │   │  │
│   │  │  │ Control          │  │                  │  │  events)     │  │   │  │
│   │  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │  │
│   │  └────────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                         │  │
│   │                              ▼                                         │  │
│   │  ┌────────────────────────────────────────────────────────────────┐   │  │
│   │  │              AUTONOMOUS CAPABILITIES                            │   │  │
│   │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │  │
│   │  │  │ Scheduled Tasks  │  │ Proactive        │  │ Multi-Account│  │   │  │
│   │  │  │ (cron-like       │  │ Messaging        │  │ Coordination │  │   │  │
│   │  │  │  background ops) │  │ (initiate convs) │  │              │  │   │  │
│   │  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │  │
│   │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │  │
│   │  │  │ Webhook/Callback │  │ External API     │  │ File System  │  │   │  │
│   │  │  │ Server           │  │ Integration      │  │ Access       │  │   │  │
│   │  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │  │
│   │  └────────────────────────────────────────────────────────────────┘   │  │
│   └───────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                    ┌────────────────────┼────────────────────┐               │
│                    │                    │                    │               │
│                    ▼                    ▼                    ▼               │
│   ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────────┐     │
│   │   delta-echo-desk   │  │    deltecho2    │  │   dovecot-core      │     │
│   │   (Desktop UI)      │  │   (Desktop UI)  │  │   (Mail Server)     │     │
│   │                     │  │                 │  │                     │     │
│   │   ┌─────────────┐   │  │ ┌─────────────┐│  │ ┌─────────────────┐ │     │
│   │   │IPC Client   │   │  │ │IPC Client   ││  │ │ MilterInterface │ │     │
│   │   │to Orchestr  │   │  │ │to Orchestr  ││  │ │ (email hooks)   │ │     │
│   │   └─────────────┘   │  │ └─────────────┘│  │ └─────────────────┘ │     │
│   └─────────────────────┘  └─────────────────┘  └─────────────────────┘     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Roadmap

### Phase 1: Extract & Consolidate (Foundation)

```
deltecho/
├── deep-tree-echo-core/           # NEW: Shared core package
│   ├── src/
│   │   ├── cognitive/
│   │   │   ├── LLMService.ts
│   │   │   ├── CognitiveCore.ts
│   │   │   ├── AffectiveCore.ts
│   │   │   └── RelevanceCore.ts
│   │   ├── memory/
│   │   │   ├── SemanticMemory.ts
│   │   │   ├── EpisodicMemory.ts
│   │   │   ├── ProceduralMemory.ts
│   │   │   ├── HyperDimensionalMemory.ts  # From deltecho2
│   │   │   └── RAGMemoryStore.ts
│   │   ├── personality/
│   │   │   ├── PersonaCore.ts
│   │   │   ├── AdaptivePersonality.ts     # From deltecho2
│   │   │   ├── EmotionalIntelligence.ts   # From deltecho2
│   │   │   └── SelfReflection.ts
│   │   ├── security/
│   │   │   └── SecureIntegration.ts       # From deltecho2
│   │   ├── embodiment/
│   │   │   └── ProprioceptiveEmbodiment.ts # From deltecho2
│   │   └── index.ts
│   └── package.json
```

### Phase 2: Orchestrator Daemon

```
deltecho/
├── deep-tree-echo-orchestrator/   # NEW: Background daemon
│   ├── src/
│   │   ├── daemon.ts              # Main process entry
│   │   ├── deltachat-interface/
│   │   │   ├── rpc-client.ts      # Direct JSON-RPC to deltachat-rpc
│   │   │   ├── account-manager.ts
│   │   │   ├── message-handler.ts
│   │   │   └── event-subscriber.ts
│   │   ├── ipc/
│   │   │   ├── server.ts          # IPC server for desktop apps
│   │   │   └── protocol.ts
│   │   ├── scheduler/
│   │   │   ├── task-scheduler.ts
│   │   │   └── cron-manager.ts
│   │   ├── webhooks/
│   │   │   └── webhook-server.ts
│   │   └── orchestrator.ts        # Main orchestration logic
│   └── package.json
```

### Phase 3: Desktop Integration

```
# delta-echo-desk & deltecho2 modifications:

packages/
├── frontend/
│   └── src/
│       └── components/
│           └── DeepTreeEchoBot/
│               ├── OrchestratorClient.ts  # NEW: IPC client
│               └── ...existing files...   # Refactored to use core
```

---

## 6. Key Capabilities for Root-Level Orchestrator

### What Deep Tree Echo Gains as System Orchestrator:

| Capability | Current | Proposed | Benefit |
|------------|---------|----------|---------|
| Message Processing | Reactive only | Proactive + Reactive | Can initiate conversations |
| Account Management | Read-only | Full CRUD | Can create bot accounts |
| Background Operation | None (UI only) | Daemon process | 24/7 operation |
| Scheduling | None | Cron-like tasks | Scheduled check-ins, reminders |
| Multi-App Coordination | Isolated | Unified | Single brain across apps |
| External Integration | Limited | Webhooks/APIs | Connect to other services |
| Mail Server Integration | None | Milter/plugin | Process emails at server level |
| Database Access | Via RPC only | Direct SQLite | Full message history |
| Protocol Control | None | IMAP/SMTP | Email transport control |

---

## 7. Communication Flow (Proposed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MESSAGE FLOW - ORCHESTRATOR MODE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INCOMING MESSAGE                                                           │
│   ================                                                           │
│                                                                              │
│   [Email Server] ──IMAP──▶ [deltachat-rpc] ──event──▶ [Orchestrator Daemon] │
│                                                              │               │
│                                              ┌───────────────┴────────────┐  │
│                                              ▼                            ▼  │
│                                   [Deep Tree Echo Core]        [IPC Broadcast]│
│                                   - Cognitive Processing       to Desktop UIs │
│                                   - Memory Retrieval                          │
│                                   - Persona Consultation                      │
│                                              │                                │
│                                              ▼                                │
│                                   [Response Generation]                       │
│                                              │                                │
│                                              ▼                                │
│   [Email Server] ◀──SMTP──  [deltachat-rpc] ◀──send──  [Orchestrator Daemon] │
│                                                                              │
│                                                                              │
│   PROACTIVE MESSAGE (NEW CAPABILITY)                                        │
│   ==================================                                        │
│                                                                              │
│   [Scheduler Triggers] ──▶ [Orchestrator Daemon]                            │
│                                     │                                        │
│                                     ▼                                        │
│                           [Deep Tree Echo Core]                              │
│                           - Determine if action needed                       │
│                           - Generate proactive message                       │
│                                     │                                        │
│                                     ▼                                        │
│   [Email Server] ◀──SMTP──  [deltachat-rpc] ◀──send──  [Orchestrator]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary: Current vs Proposed

| Aspect | Current State | Proposed State |
|--------|---------------|----------------|
| **Architecture** | Embedded in frontend (renderer process) | Separate daemon + core library |
| **Scope** | Single app instance | Monorepo-wide coordination |
| **DeltaChat Integration** | RPC via BackendRemote (limited) | Direct deltachat-rpc + DB access |
| **Autonomy** | Reactive to user messages | Proactive + scheduled + reactive |
| **Persistence** | Desktop settings (JSON) | Dedicated database + settings |
| **Cognitive Modules** | Split between repos | Unified core package |
| **dovecot Integration** | None | Milter plugin for email processing |

---

## 9. Files to Create/Modify

### New Files Needed:

1. `deltecho/deep-tree-echo-core/` - Entire new package
2. `deltecho/deep-tree-echo-orchestrator/` - Entire new package
3. `deltecho/deep-tree-echo-core/src/index.ts` - Unified exports
4. `deltecho/package.json` - Workspace configuration for monorepo

### Files to Refactor:

1. `delta-echo-desk/packages/frontend/src/components/DeepTreeEchoBot/*` - Import from core
2. `deltecho2/packages/frontend/src/components/DeepTreeEchoBot/*` - Import from core
3. Both apps' `DeepTreeEchoIntegration.ts` - Add IPC client connection

---

*This architecture positions Deep Tree Echo as a true system-level AI orchestrator, capable of operating independently while coordinating with multiple client applications and the underlying mail infrastructure.*
