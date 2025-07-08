"""Memory Agent for conversation history management."""

from typing import List, Dict, Any, Optional
import structlog
from datetime import datetime

from ..core.config import settings
from ..core.models import ChatMessage

logger = structlog.get_logger(__name__)


class MemoryAgent:
    """Agent for managing conversation history and session memory."""
    
    def __init__(self):
        """Initialize the Memory Agent."""
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.max_messages = settings.max_memory_messages
        logger.info("Memory Agent initialized", max_messages=self.max_messages)
    
    def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to the session history."""
        try:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            
            # Add the message
            self.sessions[session_id].append(message)
            
            # Maintain message limit
            if len(self.sessions[session_id]) > self.max_messages:
                # Remove oldest messages but keep the most recent ones
                self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]
            
            logger.info(
                "Added message to session",
                session_id=session_id,
                role=message.role,
                total_messages=len(self.sessions[session_id])
            )
            
            return True
            
        except Exception as e:
            logger.error("Error adding message to session", session_id=session_id, error=str(e))
            return False
    
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session."""
        try:
            if session_id in self.sessions:
                return self.sessions[session_id].copy()
            return []
            
        except Exception as e:
            logger.error("Error getting chat history", session_id=session_id, error=str(e))
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear chat history for a session."""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info("Cleared session", session_id=session_id)
                return True
            else:
                logger.warning("Session not found for clearing", session_id=session_id)
                return False
                
        except Exception as e:
            logger.error("Error clearing session", session_id=session_id, error=str(e))
            return False
    
    def get_recent_context(self, session_id: str, max_messages: int = 5) -> List[ChatMessage]:
        """Get recent messages for context."""
        try:
            history = self.get_chat_history(session_id)
            if not history:
                return []
            
            # Return the most recent messages
            return history[-max_messages:] if len(history) > max_messages else history
            
        except Exception as e:
            logger.error("Error getting recent context", session_id=session_id, error=str(e))
            return []
    
    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a user message to the session."""
        message = ChatMessage(
            role="user",
            content=content,
            metadata=metadata
        )
        return self.add_message(session_id, message)
    
    def add_assistant_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add an assistant message to the session."""
        message = ChatMessage(
            role="assistant",
            content=content,
            metadata=metadata
        )
        return self.add_message(session_id, message)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a session."""
        try:
            if session_id not in self.sessions:
                return {
                    'exists': False,
                    'message_count': 0,
                    'user_messages': 0,
                    'assistant_messages': 0
                }
            
            messages = self.sessions[session_id]
            user_count = sum(1 for msg in messages if msg.role == "user")
            assistant_count = sum(1 for msg in messages if msg.role == "assistant")
            
            # Get first and last message timestamps
            first_timestamp = messages[0].timestamp if messages else None
            last_timestamp = messages[-1].timestamp if messages else None
            
            return {
                'exists': True,
                'message_count': len(messages),
                'user_messages': user_count,
                'assistant_messages': assistant_count,
                'first_message': first_timestamp,
                'last_message': last_timestamp,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error("Error getting session stats", session_id=session_id, error=str(e))
            return {'exists': False, 'error': str(e)}
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with basic stats."""
        try:
            sessions_list = []
            for session_id in self.sessions.keys():
                stats = self.get_session_stats(session_id)
                if stats['exists']:
                    sessions_list.append(stats)
            
            return sorted(sessions_list, key=lambda x: x.get('last_message', datetime.min))
            
        except Exception as e:
            logger.error("Error listing active sessions", error=str(e))
            return []
    
    def cleanup_old_sessions(self, max_sessions: int = 100) -> int:
        """Clean up old sessions if there are too many."""
        try:
            if len(self.sessions) <= max_sessions:
                return 0
            
            # Get sessions sorted by last activity
            sessions_with_activity = []
            for session_id in self.sessions.keys():
                if self.sessions[session_id]:
                    last_activity = self.sessions[session_id][-1].timestamp
                    sessions_with_activity.append((session_id, last_activity))
            
            # Sort by last activity (oldest first)
            sessions_with_activity.sort(key=lambda x: x[1])
            
            # Remove oldest sessions
            sessions_to_remove = len(self.sessions) - max_sessions
            removed_count = 0
            
            for session_id, _ in sessions_with_activity[:sessions_to_remove]:
                del self.sessions[session_id]
                removed_count += 1
            
            logger.info("Cleaned up old sessions", removed_count=removed_count)
            return removed_count
            
        except Exception as e:
            logger.error("Error cleaning up old sessions", error=str(e))
            return 0
    
    def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get a summary of the conversation for context."""
        try:
            history = self.get_chat_history(session_id)
            if not history or len(history) < 2:
                return None
            
            # Create a simple summary of the conversation
            user_messages = [msg.content for msg in history if msg.role == "user"]
            assistant_messages = [msg.content for msg in history if msg.role == "assistant"]
            
            if not user_messages:
                return None
            
            summary = f"Conversation with {len(user_messages)} user questions"
            if len(user_messages) > 0:
                summary += f". Recent topics: {', '.join(user_messages[-3:])}"
            
            return summary
            
        except Exception as e:
            logger.error("Error getting conversation summary", session_id=session_id, error=str(e))
            return None


# Global memory agent instance
memory_agent = MemoryAgent() 