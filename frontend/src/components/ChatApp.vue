<script setup lang="ts">
import { computed, ref } from 'vue'

type ChatRole = 'user' | 'assistant' | 'system'
interface ChatMessage {
  role: ChatRole
  content: string
}

const makeSessionId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `session-${Math.random().toString(36).slice(2, 10)}`
}

const sessionId = ref(makeSessionId())
const sessionInput = ref(sessionId.value)
const query = ref('')
const messages = ref<ChatMessage[]>([])
const debugData = ref<Record<string, unknown> | null>(null)
const isLoading = ref(false)
const errorMessage = ref('')
const showDebug = ref(false)

const apiUrl = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/chat`
  : '/chat'

const historyPayload = computed(() =>
  messages.value
    .filter((m) => m.role !== 'system')
    .map((m) => `${m.role === 'assistant' ? 'Assistant' : 'User'}: ${m.content}`)
)

const sendChat = async () => {
  const prompt = query.value.trim()
  if (!prompt) {
    return
  }

  errorMessage.value = ''
  isLoading.value = true

  const userMessage: ChatMessage = {
    role: 'user',
    content: prompt,
  }

  messages.value = [...messages.value, userMessage]
  query.value = ''

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: prompt,
        session_id: sessionId.value,
        history: historyPayload.value,
      }),
    })

    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      const detail = payload?.detail || response.statusText || 'Unknown error'
      throw new Error(detail)
    }

    const data = await response.json()

    messages.value = [
      ...messages.value,
      {
        role: 'assistant',
        content: data.response || 'No response received.',
      },
    ]
    debugData.value = data.debug ?? null
    showDebug.value = true
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : 'Request failed.'
  } finally {
    isLoading.value = false
  }
}

const resetConversation = () => {
  messages.value = []
  debugData.value = null
  errorMessage.value = ''
}

const createNewSession = () => {
  sessionId.value = makeSessionId()
  sessionInput.value = sessionId.value
  resetConversation()
}

const updateSession = () => {
  const newId = sessionInput.value.trim()
  if (!newId) {
    errorMessage.value = 'Session ID cannot be empty.'
    return
  }

  if (newId !== sessionId.value) {
    sessionId.value = newId
    resetConversation()
  }
}

const handleSubmit = (event: Event) => {
  event.preventDefault()
  sendChat()
}
</script>

<template>
  <div class="app-shell">
    <header class="top-bar">
      <div>
        <strong>Multi-Agent RAG Chat</strong>
        <p>Send a question and track a session ID without login.</p>
      </div>
      <button type="button" class="secondary-button" @click="createNewSession">
        New Session
      </button>
    </header>

    <section class="session-panel">
      <div class="session-field">
        <label for="session-id">Current session ID</label>
        <input id="session-id" type="text" readonly :value="sessionId" />
      </div>
      <div class="session-field">
        <label for="session-input">Switch session ID</label>
        <input
          id="session-input"
          type="text"
          v-model="sessionInput"
          placeholder="Paste or type a session id"
        />
      </div>
      <button type="button" class="primary-button" @click="updateSession">
        Update Session
      </button>
    </section>

    <section class="chat-panel">
      <div class="chat-header">
        <div>
          <span class="badge">Live chat</span>
          <p>
            Use the active session ID for memory continuity. Updating the session clears
            conversation history locally.
          </p>
        </div>
        <span class="message-count">Messages: {{ messages.length }}</span>
      </div>

      <div class="message-list" v-if="messages.length > 0">
        <div
          v-for="(message, index) in messages"
          :key="`${message.role}-${index}`"
          :class="['message-row', message.role === 'assistant' ? 'assistant' : 'user']"
        >
          <div class="message-label">{{ message.role === 'assistant' ? 'Assistant' : 'You' }}</div>
          <p>{{ message.content }}</p>
        </div>
      </div>

      <div class="empty-state" v-else>
        <p>Enter a question below to start a session.</p>
      </div>

      <form class="prompt-form" @submit.prevent="handleSubmit">
        <textarea
          v-model="query"
          rows="4"
          placeholder="Ask your agent anything..."
          :disabled="isLoading"
        />
        <div class="form-actions">
          <button type="submit" class="primary-button" :disabled="isLoading || !query.trim()">
            {{ isLoading ? 'Sending...' : 'Send message' }}
          </button>
          <button type="button" class="secondary-button" @click="resetConversation" :disabled="isLoading">
            Clear chat
          </button>
        </div>
      </form>

      <div class="error-banner" v-if="errorMessage">
        {{ errorMessage }}
      </div>

      <div class="debug-panel" v-if="debugData">
        <div class="debug-header" @click="showDebug = !showDebug">
          <span>Debug payload</span>
          <span>{{ showDebug ? '▲' : '▼' }}</span>
        </div>
        <pre v-if="showDebug">{{ JSON.stringify(debugData, null, 2) }}</pre>
      </div>
    </section>
  </div>
</template>

<style scoped>
.app-shell {
  max-width: 960px;
  margin: 0 auto;
  padding: 1.5rem;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: #f3f4f6;
}

.top-bar {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  padding: 1rem 1.25rem;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
}

.top-bar strong {
  display: block;
  font-size: 1.2rem;
  margin-bottom: 0.25rem;
}

.top-bar p {
  margin: 0;
  color: #cbd5e1;
}

.session-panel {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 1rem;
  align-items: end;
  margin: 1.5rem 0;
}

.session-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.session-field label {
  font-size: 0.9rem;
  color: #94a3b8;
}

.session-field input {
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.9);
  padding: 0.85rem 1rem;
  color: #e2e8f0;
}

.chat-panel {
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.12);
  border-radius: 24px;
  padding: 1.5rem;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  background: #0f172a;
  color: #38bdf8;
  font-size: 0.85rem;
  letter-spacing: 0.01em;
}

.message-count {
  color: #94a3b8;
  font-size: 0.9rem;
}

.message-list {
  display: grid;
  gap: 0.95rem;
  margin-bottom: 1.5rem;
}

.message-row {
  padding: 1rem 1.1rem;
  border-radius: 20px;
  line-height: 1.6;
  max-width: 100%;
}

.message-row.user {
  background: rgba(59, 130, 246, 0.16);
  color: #c7d2fe;
  align-self: flex-end;
}

.message-row.assistant {
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.2);
}

.message-label {
  font-size: 0.78rem;
  color: #94a3b8;
  margin-bottom: 0.35rem;
}

.empty-state {
  padding: 1.5rem;
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.7);
  border: 1px dashed rgba(148, 163, 184, 0.35);
  color: #cbd5e1;
  text-align: center;
  margin-bottom: 1.5rem;
}

.prompt-form {
  display: grid;
  gap: 1rem;
}

.prompt-form textarea {
  width: 100%;
  min-height: 130px;
  resize: vertical;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.92);
  color: #e2e8f0;
  padding: 1rem;
  font-size: 1rem;
}

.form-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.primary-button,
.secondary-button {
  border: none;
  border-radius: 14px;
  padding: 0.95rem 1.2rem;
  cursor: pointer;
  font-weight: 600;
}

.primary-button {
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  color: #0f172a;
}

.secondary-button {
  background: rgba(148, 163, 184, 0.15);
  color: #cbd5e1;
}

.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.error-banner {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border-radius: 16px;
  background: rgba(248, 113, 113, 0.12);
  color: #fecaca;
}

.debug-panel {
  margin-top: 1.5rem;
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.debug-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.1rem;
  cursor: pointer;
  color: #cbd5e1;
}

.debug-panel pre {
  margin: 0;
  padding: 1rem 1.1rem;
  overflow-x: auto;
  color: #e2e8f0;
  background: rgba(15, 23, 42, 0.95);
}
</style>
