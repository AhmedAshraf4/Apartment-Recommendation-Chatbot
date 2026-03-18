import { useEffect, useMemo, useRef, useState } from "react";
import PageHeader from "../components/PageHeader";
import { API_BASE } from "../api";

function createSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export default function UserPage() {
  const sessionId = useMemo(() => createSessionId(), []);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to Dorra Real Estate Assistant. Tell me what kind of property you’re looking for, and I’ll help you find matching options.",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (event) => {
    event.preventDefault();

    const trimmedMessage = inputValue.trim();
    if (!trimmedMessage || isLoading) {
      return;
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      { role: "user", content: trimmedMessage },
      { role: "assistant", content: "Thinking..." },
    ]);

    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: trimmedMessage,
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
      let streamedText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        streamedText += chunk;

        setMessages((currentMessages) => {
          const nextMessages = [...currentMessages];
          nextMessages[nextMessages.length - 1] = {
            role: "assistant",
            content: streamedText,
          };
          return nextMessages;
        });
      }

      if (!streamedText.trim()) {
        setMessages((currentMessages) => {
          const nextMessages = [...currentMessages];
          nextMessages[nextMessages.length - 1] = {
            role: "assistant",
            content: "Sorry, I couldn’t generate a response.",
          };
          return nextMessages;
        });
      }
    } catch (error) {
      setMessages((currentMessages) => {
        const nextMessages = [...currentMessages];
        nextMessages[nextMessages.length - 1] = {
          role: "assistant",
          content:
            error.message || "Something went wrong while contacting the server.",
        };
        return nextMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <PageHeader
        title="User Chat"
        subtitle="Chat with the assistant to find matching Dorra properties."
      />

      <div className="chat-page">
        <div className="chat-card">
          <div className="chat-messages" ref={messagesContainerRef}>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`chat-bubble ${
                  message.role === "user"
                    ? "chat-bubble--user"
                    : "chat-bubble--assistant"
                }`}
              >
                <div className="chat-bubble__label">
                  {message.role === "user" ? "You" : "Assistant"}
                </div>

                <div className="chat-bubble__content">
                  {message.content}
                  {isLoading && index === messages.length - 1 && (
                    <span className="typing-cursor"></span>
                  )}
                </div>
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input-area" onSubmit={handleSend}>
            <textarea
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              placeholder="Describe the property you want..."
              rows={3}
            />
            <button className="btn btn-primary" type="submit" disabled={isLoading}>
              {isLoading ? "Streaming..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}