import { useEffect, useMemo, useRef, useState } from "react";
import PageHeader from "../components/PageHeader";
import { API_BASE } from "../api";

function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export default function UserPage() {
  const sessionId = useMemo(() => generateSessionId(), []);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to Dorra Real Estate Assistant. Tell me what kind of property you’re looking for, and I’ll help you find matching options.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e) {
    e.preventDefault();

    const message = input.trim();
    if (!message || loading) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: message },
      { role: "assistant", content: "Thinking..." },
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message,
        }),
        credentials: "include",
      });

      if (!response.ok) {
        let errorMessage = "Something went wrong while contacting the server.";
        try {
          const data = await response.json();
          errorMessage = data.detail || errorMessage;
        } catch {}
        throw new Error(errorMessage);
      }

      if (!response.body) {
        throw new Error("Streaming is not supported by this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: fullText,
          };
          return updated;
        });
      }

      if (!fullText.trim()) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: "Sorry, I couldn’t generate a response.",
          };
          return updated;
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content:
            err.message || "Something went wrong while contacting the server.",
        };
        return updated;
      });
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
          <div className="chat-messages" ref={messagesContainerRef}>
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`chat-bubble ${
                  msg.role === "user"
                    ? "chat-bubble--user"
                    : "chat-bubble--assistant"
                }`}
              >
                <div className="chat-bubble__label">
                  {msg.role === "user" ? "You" : "Assistant"}
                </div>
                <div className="chat-bubble__content">
                  {msg.content}
                  {loading && index === messages.length - 1 && (
                    <span className="typing-cursor"></span>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input-area" onSubmit={handleSend}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe the property you want..."
              rows={3}
            />
            <button className="btn btn-primary" type="submit" disabled={loading}>
              {loading ? "Streaming..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}