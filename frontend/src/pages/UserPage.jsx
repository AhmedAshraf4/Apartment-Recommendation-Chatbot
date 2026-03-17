import { useMemo, useState } from "react";
import PageHeader from "../components/PageHeader";
import { apiRequest } from "../api";

function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export default function UserPage() {
  const sessionId = useMemo(() => generateSessionId(), []);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to Dorra Real Estate Assistant. Tell me what kind of property you’re looking for, and I’ll help you find matching options.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend(e) {
    e.preventDefault();

    const message = input.trim();
    if (!message || loading) return;

    const userMessage = { role: "user", content: message };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const data = await apiRequest("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message,
        }),
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.reply || "Sorry, I couldn’t generate a response.",
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err.message || "Something went wrong while contacting the server.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <PageHeader
        title="User Chat"
        subtitle="Chat with the assistant to find matching Dorra properties."
      />

      <div className="chat-page">
        <div className="chat-card">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`chat-bubble ${
                  msg.role === "user" ? "chat-bubble--user" : "chat-bubble--assistant"
                }`}
              >
                <div className="chat-bubble__label">
                  {msg.role === "user" ? "You" : "Assistant"}
                </div>
                <div className="chat-bubble__content">{msg.content}</div>
              </div>
            ))}

            {loading && (
              <div className="chat-bubble chat-bubble--assistant">
                <div className="chat-bubble__label">Assistant</div>
                <div className="chat-bubble__content">Typing...</div>
              </div>
            )}
          </div>

          <form className="chat-input-area" onSubmit={handleSend}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe the property you want..."
              rows={3}
            />
            <button className="btn btn-primary" type="submit" disabled={loading}>
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}