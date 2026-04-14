import { useEffect, useMemo, useRef, useState } from "react";
import PageHeader from "../components/PageHeader";
import { API_BASE } from "../api";

function createSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function parseSSEChunk(chunk, onEvent) {
  const events = chunk.split("\n\n");

  for (const rawEvent of events) {
    const trimmed = rawEvent.trim();
    if (!trimmed) continue;

    const lines = trimmed.split("\n");
    let eventName = "message";
    let dataText = "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataText += line.slice(5).trim();
      }
    }

    if (!dataText) continue;

    try {
      const parsed = JSON.parse(dataText);
      onEvent(eventName, parsed);
    } catch {
      // ignore malformed event payloads
    }
  }
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
      { role: "assistant", content: "" },
    ]);

    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
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
      let buffer = "";
      let streamedText = "";
      let gotAnyToken = false;

      while (true) {
        const { value, done } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          parseSSEChunk(part + "\n\n", (eventName, payload) => {
            if (eventName === "token" && payload?.token) {
              gotAnyToken = true;
              streamedText += payload.token;

              setMessages((currentMessages) => {
                const nextMessages = [...currentMessages];
                nextMessages[nextMessages.length - 1] = {
                  role: "assistant",
                  content: streamedText,
                };
                return nextMessages;
              });
            }

            if (eventName === "error") {
              throw new Error(payload?.error || "Streaming failed.");
            }
          });
        }
      }

      if (buffer.trim()) {
        parseSSEChunk(buffer, (eventName, payload) => {
          if (eventName === "token" && payload?.token) {
            gotAnyToken = true;
            streamedText += payload.token;
          }
        });

        setMessages((currentMessages) => {
          const nextMessages = [...currentMessages];
          nextMessages[nextMessages.length - 1] = {
            role: "assistant",
            content: streamedText,
          };
          return nextMessages;
        });
      }

      if (!gotAnyToken || !streamedText.trim()) {
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