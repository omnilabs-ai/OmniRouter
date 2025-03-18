"""
Session Tracker for SmartRouter

This module tracks user sessions and collects implicit feedback data for the
reinforcement learning system. It captures metrics like:

1. Time spent reading responses
2. Follow-up questions
3. Content usage (copying, highlighting)
4. Session length and engagement

The data is used to generate implicit feedback signals that help train the
RL model even when users don't provide explicit ratings.
"""

import time
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading

class SessionData:
    """
    Represents data for a single user session.
    
    Tracks user interactions and engagement metrics to derive implicit feedback.
    """
    
    def __init__(self, session_id: str = None, user_id: Optional[str] = None):
        """
        Initialize a new session.
        
        Args:
            session_id: Unique identifier for the session (generated if not provided)
            user_id: Optional identifier for the user
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.start_time = time.time()
        
        # Basic session metrics
        self.messages = []  # List of message events
        self.interactions = []  # List of user interactions
        
        # Engagement metrics
        self.time_between_messages = []  # Time between messages in seconds
        
        # Last activity time
        self.last_activity_time = time.time()
        
        # Message analysis
        self.response_view_times = {}  # Maps message_id to view time in seconds
        self.current_message_view = None  # Current message being viewed
        self.current_view_start_time = None  # When current message view started
        
        # Model selection tracking
        self.model_selections = {}  # Maps query_id to model selection info
        
        # Engagement metrics
        self.message_view_times = {}  # Message ID -> time spent viewing
        self.content_copied = set()  # Message IDs of copied content
        self.content_highlighted = set()  # Message IDs of highlighted content
        
        # Query-response tracking
        self.queries = {}  # Query ID -> query data
        self.responses = {}  # Response ID -> response data
        
        # Current state
        self.current_message_id = None
        self.current_message_start_time = None
    
    def add_message(self, 
                   message_id: str, 
                   content: Any, 
                   role: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a new message in the session.
        
        Args:
            message_id: Unique identifier for the message
            content: Content of the message (will be converted to string if not already)
            role: Role of the message sender (user, assistant)
            metadata: Additional metadata about the message
        """
        if self.current_message_id:
            self._end_current_message_view()
        
        # Ensure content is a string
        content_str = str(content) if not isinstance(content, str) else content
        
        message_data = {
            "message_id": message_id,
            "content": content_str,
            "role": role,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "length": len(content_str),
            "word_count": len(content_str.split())
        }
        
        self.messages.append(message_data)
        
        # If this is a user message, record it as a query
        if role.lower() == "user":
            self.queries[message_id] = {
                **message_data,
                "has_response": False,
                "response_time": None
            }
        
        # If this is an assistant message, record it as a response
        # and link it to the most recent query
        elif role.lower() == "assistant":
            self.responses[message_id] = message_data
            
            # Find the most recent query without a response
            for query_id, query_data in reversed(list(self.queries.items())):
                if not query_data["has_response"]:
                    query_data["has_response"] = True
                    query_data["response_time"] = time.time() - query_data["timestamp"]
                    query_data["response_id"] = message_id
                    break
        
        # Start tracking time for this message
        self.current_message_id = message_id
        self.current_message_start_time = time.time()
    
    def _end_current_message_view(self) -> None:
        """End the current message view and update view time"""
        if self.current_message_id and self.current_view_start_time:
            view_time = time.time() - self.current_view_start_time
            self.message_view_times[self.current_message_id] = self.message_view_times.get(self.current_message_id, 0) + view_time
            self.current_message_id = None
            self.current_view_start_time = None
    
    def _count_follow_up_questions(self) -> int:
        """Count the number of follow-up questions in the conversation"""
        follow_ups = 0
        if len(self.messages) >= 2:
            for i in range(1, len(self.messages)):
                if (self.messages[i]["role"].lower() == "user" and 
                    self.messages[i-1]["role"].lower() == "assistant"):
                    follow_ups += 1
        return follow_ups
    
    def record_copy(self, message_id: str) -> None:
        """
        Record that content from a message was copied.
        
        Args:
            message_id: ID of the message that was copied
        """
        self.content_copied.add(message_id)
        
        self.interactions.append({
            "type": "copy",
            "message_id": message_id,
            "timestamp": time.time()
        })
    
    def record_highlight(self, message_id: str) -> None:
        """
        Record that content from a message was highlighted.
        
        Args:
            message_id: ID of the message that was highlighted
        """
        self.content_highlighted.add(message_id)
        
        self.interactions.append({
            "type": "highlight",
            "message_id": message_id,
            "timestamp": time.time()
        })
    
    def record_model_selection(self, query_id: str, model_name: str, scores: Dict[str, float]) -> None:
        """
        Record a model selection for a query.
        
        Args:
            query_id: ID of the query message
            model_name: Name of the selected model
            scores: Scores for different models
        """
        self.model_selections[query_id] = {
            "model_name": model_name,
            "scores": scores,
            "timestamp": time.time()
        }
    
    def record_interaction(self, interaction_type: str, data: Dict[str, Any] = None) -> None:
        """
        Record a general user interaction.
        
        Args:
            interaction_type: Type of interaction
            data: Additional data about the interaction
        """
        self.interactions.append({
            "type": interaction_type,
            "data": data or {},
            "timestamp": time.time()
        })
    
    def get_engagement_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about user engagement in this session.
        
        Returns:
            Dictionary of engagement metrics
        """
        # Finalize current message view if there is one
        if self.current_message_id:
            self._end_current_message_view()
        
        # Calculate basic metrics
        session_length = time.time() - self.start_time
        message_count = len(self.messages)
        user_message_count = sum(1 for m in self.messages if m["role"].lower() == "user")
        assistant_message_count = sum(1 for m in self.messages if m["role"].lower() == "assistant")
        
        # Calculate average time viewing assistant messages
        assistant_view_times = [
            self.message_view_times.get(m["message_id"], 0)
            for m in self.messages
            if m["role"].lower() == "assistant" and m["message_id"] in self.message_view_times
        ]
        
        avg_view_time = sum(assistant_view_times) / max(1, len(assistant_view_times))
        
        # Calculate time per word for assistant messages
        time_per_word_values = []
        for m in self.messages:
            if m["role"].lower() == "assistant" and m["message_id"] in self.message_view_times:
                view_time = self.message_view_times.get(m["message_id"], 0)
                word_count = m["word_count"]
                if word_count > 0:
                    time_per_word_values.append(view_time / word_count)
        
        avg_time_per_word = sum(time_per_word_values) / max(1, len(time_per_word_values))
        
        # Calculate content engagement
        content_copied_ratio = len(self.content_copied) / max(1, assistant_message_count)
        content_highlighted_ratio = len(self.content_highlighted) / max(1, assistant_message_count)
        
        # Check for follow-up questions
        follow_ups = self._count_follow_up_questions()
        
        return {
            "session_length": session_length,
            "message_count": message_count,
            "user_message_count": user_message_count,
            "assistant_message_count": assistant_message_count,
            "avg_assistant_view_time_seconds": avg_view_time,
            "avg_time_per_word_seconds": avg_time_per_word,
            "content_copied_ratio": content_copied_ratio,
            "content_highlighted_ratio": content_highlighted_ratio,
            "follow_up_questions": follow_ups,
            "time_reading_seconds": sum(assistant_view_times),
            "copied_content": len(self.content_copied) > 0,
            "highlighted_content": len(self.content_highlighted) > 0,
            "interaction_count": len(self.interactions)
        }
    
    def get_implicit_feedback_data(self) -> Dict[str, Any]:
        """
        Get data for implicit feedback calculation.
        
        Returns:
            Dictionary of session data for implicit feedback
        """
        engagement_metrics = self.get_engagement_metrics()
        
        # Include data about the last query-response pair
        last_query = None
        last_response = None
        
        if self.messages:
            # Find the last query-response pair
            for i in range(len(self.messages) - 1, 0, -1):
                if (self.messages[i]["role"].lower() == "assistant" and
                    self.messages[i-1]["role"].lower() == "user"):
                    last_response = self.messages[i]
                    last_query = self.messages[i-1]
                    break
        
        # Get model selection for last query if available
        model_selection = None
        if last_query and last_query["message_id"] in self.model_selections:
            model_selection = self.model_selections[last_query["message_id"]]
        
        # Calculate response-specific metrics if available
        response_metrics = {}
        if last_response and last_response["message_id"] in self.message_view_times:
            response_id = last_response["message_id"]
            response_metrics = {
                "view_time_seconds": self.message_view_times.get(response_id, 0),
                "was_copied": response_id in self.content_copied,
                "was_highlighted": response_id in self.content_highlighted,
                "word_count": last_response["word_count"],
                "time_per_word": (self.message_view_times.get(response_id, 0) / 
                                 max(1, last_response["word_count"]))
            }
        
        # Calculate a simple engagement score for testing
        engagement_score = 0.0
        if engagement_metrics:
            # Base score on a few key metrics
            score_components = [
                0.5,  # Base score
                0.1 if engagement_metrics.get("message_count", 0) > 1 else 0,
                0.2 if engagement_metrics.get("follow_up_questions", 0) > 0 else 0,
                0.1 if engagement_metrics.get("copied_content", False) else 0,
                0.1 if engagement_metrics.get("highlighted_content", False) else 0
            ]
            engagement_score = sum(score_components)
        
        return {
            "session_id": self.session_id,
            "session_data": engagement_metrics,
            "last_query": last_query,
            "last_response": last_response,
            "model_selection": model_selection,
            "response_metrics": response_metrics,
            "engagement_score": engagement_score
        }


class SessionTracker:
    """
    Tracks multiple user sessions and collects data for implicit feedback.
    """
    
    def __init__(self, session_timeout_seconds: int = 30 * 60):
        """
        Initialize the session tracker.
        
        Args:
            session_timeout_seconds: Time in seconds after which a session is considered inactive
        """
        self.sessions = {}  # Session ID -> SessionData
        self.session_timeout_seconds = session_timeout_seconds
        self.lock = threading.RLock()  # Thread-safe operations
        
        # Start a background thread to clean up inactive sessions
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def get_session(self, session_id: str, user_id: Optional[str] = None) -> SessionData:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Unique identifier for the session
            user_id: Optional identifier for the user
            
        Returns:
            SessionData object for the session
        """
        with self.lock:
            # Check if session exists and is active
            if session_id in self.sessions:
                return self.sessions[session_id]
            
            # Create new session
            session = SessionData(session_id=session_id, user_id=user_id)
            self.sessions[session_id] = session
            return session
    
    def record_message(self, 
                      session_id: str, 
                      message_id: str, 
                      content: Any, 
                      role: str, 
                      user_id: Optional[str] = None) -> None:
        """
        Record a message in a session.
        
        Args:
            session_id: Unique identifier for the session
            message_id: Unique identifier for the message
            content: Content of the message (will be converted to string if not already)
            role: Role of the message sender (user, assistant)
            user_id: Optional identifier for the user
        """
        session = self.get_session(session_id, user_id)
        session.add_message(message_id, content, role)
    
    def record_model_selection(self, 
                              session_id: str, 
                              query_id: str, 
                              model_name: str, 
                              scores: Dict[str, float]) -> None:
        """
        Record a model selection in a session.
        
        Args:
            session_id: Unique identifier for the session
            query_id: ID of the query message
            model_name: Name of the selected model
            scores: Scores for different models
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].record_model_selection(query_id, model_name, scores)
    
    def record_interaction(self, 
                          session_id: str, 
                          interaction_type: str, 
                          data: Dict[str, Any] = None) -> None:
        """
        Record a user interaction in a session.
        
        Args:
            session_id: Unique identifier for the session
            interaction_type: Type of interaction
            data: Additional data about the interaction
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].record_interaction(interaction_type, data)
    
    def record_copy(self, session_id: str, message_id: str) -> None:
        """
        Record that content from a message was copied.
        
        Args:
            session_id: Unique identifier for the session
            message_id: ID of the message that was copied
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].record_copy(message_id)
    
    def record_highlight(self, session_id: str, message_id: str) -> None:
        """
        Record that content from a message was highlighted.
        
        Args:
            session_id: Unique identifier for the session
            message_id: ID of the message that was highlighted
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].record_highlight(message_id)
    
    def get_engagement_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get engagement metrics for a session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary of engagement metrics
        """
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].get_engagement_metrics()
            return {}
    
    def get_implicit_feedback_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get data for implicit feedback calculation for a session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Dictionary of session data for implicit feedback
        """
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id].get_implicit_feedback_data()
            return {}
    
    def _cleanup_worker(self) -> None:
        """Background worker that cleans up inactive sessions."""
        while True:
            try:
                self._cleanup_inactive_sessions()
            except Exception as e:
                print(f"Error cleaning up sessions: {e}")
            
            # Sleep for 5 minutes before next cleanup
            time.sleep(300)
    
    def _cleanup_inactive_sessions(self) -> None:
        """Remove sessions that have been inactive for too long."""
        current_time = time.time()
        inactive_sessions = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                # If the session has messages, check the time of the last message
                if session.messages:
                    last_activity = session.messages[-1]["timestamp"]
                else:
                    last_activity = session.start_time
                
                # If the session has been inactive for too long, mark it for removal
                if current_time - last_activity > self.session_timeout_seconds:
                    inactive_sessions.append(session_id)
            
            # Remove inactive sessions
            for session_id in inactive_sessions:
                del self.sessions[session_id]
                
        if inactive_sessions:
            print(f"Cleaned up {len(inactive_sessions)} inactive sessions")


# Global instance for session tracking
SESSION_TRACKER = SessionTracker() 