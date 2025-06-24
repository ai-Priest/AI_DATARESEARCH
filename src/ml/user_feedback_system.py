"""
User Feedback System for Continuous ML Improvement
Captures user interactions and preferences to improve recommendations over time.
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class UserFeedbackCollector:
    """Collects and processes user feedback for ML model improvement."""
    
    def __init__(self, feedback_file: str = "data/feedback/user_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = []
        self.user_preferences = defaultdict(dict)
        self.query_performance = defaultdict(list)
        self.session_data = defaultdict(dict)
        
        # Ensure feedback directory exists
        Path(feedback_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        self._load_existing_feedback()
    
    def _load_existing_feedback(self):
        """Load existing feedback data from file."""
        try:
            if Path(self.feedback_file).exists():
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.feedback_data)} existing feedback entries")
            else:
                logger.info("📝 Starting with fresh feedback system")
        except Exception as e:
            logger.warning(f"Could not load existing feedback: {e}")
            self.feedback_data = []
    
    def record_search_interaction(self, user_id: str, session_id: str, 
                                query: str, results: List[Dict], 
                                interaction_type: str, dataset_id: str = None,
                                rating: int = None, timestamp: datetime = None) -> str:
        """
        Record a user interaction with search results.
        
        Args:
            user_id: Unique user identifier
            session_id: Session identifier
            query: User's search query
            results: List of recommended datasets
            interaction_type: 'click', 'download', 'bookmark', 'rating', 'ignore'
            dataset_id: ID of interacted dataset (if applicable)
            rating: User rating 1-5 (if applicable)
            timestamp: When interaction occurred
            
        Returns:
            Feedback entry ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        feedback_entry = {
            'id': f"{user_id}_{session_id}_{timestamp.timestamp()}",
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': timestamp.isoformat(),
            'query': query,
            'interaction_type': interaction_type,
            'dataset_id': dataset_id,
            'rating': rating,
            'results_shown': len(results),
            'result_titles': [r.get('title', '') for r in results[:5]],  # Top 5 for analysis
            'result_scores': [r.get('score', 0) for r in results[:5]]
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Update session tracking
        if session_id not in self.session_data[user_id]:
            self.session_data[user_id][session_id] = {
                'start_time': timestamp.isoformat(),
                'queries': [],
                'interactions': [],
                'total_results_shown': 0
            }
        
        self.session_data[user_id][session_id]['queries'].append(query)
        self.session_data[user_id][session_id]['interactions'].append(interaction_type)
        self.session_data[user_id][session_id]['total_results_shown'] += len(results)
        
        logger.info(f"📝 Recorded {interaction_type} interaction for user {user_id}")
        return feedback_entry['id']
    
    def record_query_expansion_feedback(self, user_id: str, session_id: str,
                                      original_query: str, expanded_query: str,
                                      expansion_helpful: bool, 
                                      preferred_terms: List[str] = None):
        """Record feedback on query expansion effectiveness."""
        
        feedback_entry = {
            'id': f"expansion_{user_id}_{session_id}_{datetime.now().timestamp()}",
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'query_expansion',
            'original_query': original_query,
            'expanded_query': expanded_query,
            'expansion_helpful': expansion_helpful,
            'preferred_terms': preferred_terms or [],
            'rejected_terms': []
        }
        
        # Extract rejected terms (in expanded but not preferred)
        if preferred_terms:
            expanded_terms = set(expanded_query.split()) - set(original_query.split())
            preferred_set = set(preferred_terms)
            feedback_entry['rejected_terms'] = list(expanded_terms - preferred_set)
        
        self.feedback_data.append(feedback_entry)
        logger.info(f"📝 Recorded query expansion feedback for user {user_id}")
    
    def record_recommendation_quality(self, user_id: str, session_id: str,
                                    query: str, recommendations: List[Dict],
                                    overall_satisfaction: int,
                                    relevance_ratings: List[int] = None):
        """Record overall satisfaction with recommendation quality."""
        
        feedback_entry = {
            'id': f"quality_{user_id}_{session_id}_{datetime.now().timestamp()}",
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'recommendation_quality',
            'query': query,
            'overall_satisfaction': overall_satisfaction,  # 1-5 scale
            'num_recommendations': len(recommendations),
            'relevance_ratings': relevance_ratings or [],
            'recommendation_titles': [r.get('title', '') for r in recommendations[:5]]
        }
        
        self.feedback_data.append(feedback_entry)
        logger.info(f"📝 Recorded recommendation quality feedback: {overall_satisfaction}/5")
    
    def analyze_user_preferences(self, user_id: str) -> Dict:
        """Analyze user preferences from feedback history."""
        user_feedback = [f for f in self.feedback_data if f['user_id'] == user_id]
        
        if not user_feedback:
            return {'message': 'No feedback data available for user'}
        
        analysis = {
            'total_interactions': len(user_feedback),
            'interaction_types': defaultdict(int),
            'preferred_categories': defaultdict(int),
            'query_patterns': defaultdict(int),
            'satisfaction_trend': [],
            'click_through_rate': 0,
            'avg_session_length': 0
        }
        
        # Analyze interaction types
        for feedback in user_feedback:
            interaction_type = feedback.get('interaction_type', 'unknown')
            analysis['interaction_types'][interaction_type] += 1
            
            # Track satisfaction over time
            if feedback.get('rating'):
                analysis['satisfaction_trend'].append({
                    'timestamp': feedback['timestamp'],
                    'rating': feedback['rating']
                })
            
            # Extract query patterns
            if 'query' in feedback:
                query_words = feedback['query'].lower().split()
                for word in query_words:
                    if len(word) > 3:  # Skip short words
                        analysis['query_patterns'][word] += 1
        
        # Calculate click-through rate
        total_searches = sum(1 for f in user_feedback if 'results_shown' in f)
        total_clicks = analysis['interaction_types']['click']
        if total_searches > 0:
            analysis['click_through_rate'] = total_clicks / total_searches
        
        # Convert defaultdicts to regular dicts for JSON serialization
        analysis['interaction_types'] = dict(analysis['interaction_types'])
        analysis['preferred_categories'] = dict(analysis['preferred_categories'])
        analysis['query_patterns'] = dict(analysis['query_patterns'])
        
        return analysis
    
    def get_personalization_insights(self, user_id: str) -> Dict:
        """Generate personalization insights for improving recommendations."""
        preferences = self.analyze_user_preferences(user_id)
        
        insights = {
            'recommended_adjustments': [],
            'preferred_query_terms': [],
            'disliked_patterns': [],
            'engagement_score': 0
        }
        
        if preferences['total_interactions'] == 0:
            return insights
        
        # Extract top preferred terms
        query_patterns = preferences.get('query_patterns', {})
        insights['preferred_query_terms'] = sorted(
            query_patterns.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Calculate engagement score
        interactions = preferences['interaction_types']
        positive_interactions = interactions.get('click', 0) + interactions.get('download', 0) + interactions.get('bookmark', 0)
        total_interactions = sum(interactions.values())
        
        if total_interactions > 0:
            insights['engagement_score'] = positive_interactions / total_interactions
        
        # Generate recommendations
        if insights['engagement_score'] < 0.3:
            insights['recommended_adjustments'].append('Improve result relevance - low engagement detected')
        
        if preferences['click_through_rate'] < 0.1:
            insights['recommended_adjustments'].append('Results may not match user intent - expand query context')
        
        if len(preferences['satisfaction_trend']) > 3:
            recent_ratings = [s['rating'] for s in preferences['satisfaction_trend'][-3:]]
            if np.mean(recent_ratings) < 3:
                insights['recommended_adjustments'].append('Recent satisfaction declining - review recommendation algorithm')
        
        return insights
    
    def generate_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate training data from user feedback for model improvement.
        
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        positive_examples = []
        negative_examples = []
        
        for feedback in self.feedback_data:
            if 'query' not in feedback or 'interaction_type' not in feedback:
                continue
            
            example = {
                'query': feedback['query'],
                'timestamp': feedback['timestamp'],
                'user_id': feedback['user_id']
            }
            
            # Add result information if available
            if 'result_titles' in feedback:
                example['results'] = feedback['result_titles']
            if 'result_scores' in feedback:
                example['scores'] = feedback['result_scores']
            
            # Classify as positive or negative based on interaction
            interaction_type = feedback['interaction_type']
            
            if interaction_type in ['click', 'download', 'bookmark'] or \
               (feedback.get('rating') and feedback['rating'] >= 4):
                positive_examples.append(example)
            elif interaction_type in ['ignore', 'back'] or \
                 (feedback.get('rating') and feedback['rating'] <= 2):
                negative_examples.append(example)
        
        logger.info(f"📊 Generated {len(positive_examples)} positive and {len(negative_examples)} negative examples")
        return positive_examples, negative_examples
    
    def save_feedback(self):
        """Save feedback data to file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2, default=str)
            logger.info(f"💾 Saved {len(self.feedback_data)} feedback entries")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def get_feedback_summary(self) -> Dict:
        """Get summary statistics of collected feedback."""
        if not self.feedback_data:
            return {'message': 'No feedback data available'}
        
        summary = {
            'total_feedback_entries': len(self.feedback_data),
            'unique_users': len(set(f['user_id'] for f in self.feedback_data)),
            'unique_sessions': len(set(f['session_id'] for f in self.feedback_data)),
            'interaction_breakdown': defaultdict(int),
            'average_satisfaction': 0,
            'feedback_timespan': None
        }
        
        ratings = []
        timestamps = []
        
        for feedback in self.feedback_data:
            # Count interaction types
            interaction_type = feedback.get('interaction_type', 'unknown')
            summary['interaction_breakdown'][interaction_type] += 1
            
            # Collect ratings
            if feedback.get('rating'):
                ratings.append(feedback['rating'])
            
            # Collect timestamps
            if feedback.get('timestamp'):
                timestamps.append(feedback['timestamp'])
        
        # Calculate averages
        if ratings:
            summary['average_satisfaction'] = np.mean(ratings)
        
        # Calculate timespan
        if timestamps:
            timestamps.sort()
            summary['feedback_timespan'] = {
                'start': timestamps[0],
                'end': timestamps[-1],
                'duration_days': (datetime.fromisoformat(timestamps[-1]) - 
                                datetime.fromisoformat(timestamps[0])).days
            }
        
        # Convert defaultdict for JSON serialization
        summary['interaction_breakdown'] = dict(summary['interaction_breakdown'])
        
        return summary


class FeedbackDrivenModelImprover:
    """Uses feedback data to suggest and implement model improvements."""
    
    def __init__(self, feedback_collector: UserFeedbackCollector):
        self.feedback_collector = feedback_collector
    
    def analyze_query_expansion_performance(self) -> Dict:
        """Analyze how well query expansion is working based on user feedback."""
        expansion_feedback = [
            f for f in self.feedback_collector.feedback_data 
            if f.get('feedback_type') == 'query_expansion'
        ]
        
        if not expansion_feedback:
            return {'message': 'No query expansion feedback available'}
        
        analysis = {
            'total_expansion_feedback': len(expansion_feedback),
            'helpful_rate': 0,
            'commonly_preferred_terms': defaultdict(int),
            'commonly_rejected_terms': defaultdict(int),
            'suggestions': []
        }
        
        helpful_count = 0
        for feedback in expansion_feedback:
            if feedback.get('expansion_helpful'):
                helpful_count += 1
                
                # Count preferred terms
                for term in feedback.get('preferred_terms', []):
                    analysis['commonly_preferred_terms'][term] += 1
            else:
                # Count rejected terms
                for term in feedback.get('rejected_terms', []):
                    analysis['commonly_rejected_terms'][term] += 1
        
        if len(expansion_feedback) > 0:
            analysis['helpful_rate'] = helpful_count / len(expansion_feedback)
        
        # Generate suggestions
        if analysis['helpful_rate'] < 0.5:
            analysis['suggestions'].append('Query expansion needs improvement - consider reducing expansion aggressiveness')
        
        if analysis['commonly_rejected_terms']:
            top_rejected = sorted(analysis['commonly_rejected_terms'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            analysis['suggestions'].append(f'Consider removing these frequently rejected terms: {[t[0] for t in top_rejected]}')
        
        # Convert defaultdicts
        analysis['commonly_preferred_terms'] = dict(analysis['commonly_preferred_terms'])
        analysis['commonly_rejected_terms'] = dict(analysis['commonly_rejected_terms'])
        
        return analysis
    
    def suggest_model_improvements(self) -> List[Dict]:
        """Generate specific suggestions for model improvements based on feedback."""
        suggestions = []
        
        # Analyze overall satisfaction
        quality_feedback = [
            f for f in self.feedback_collector.feedback_data 
            if f.get('feedback_type') == 'recommendation_quality'
        ]
        
        if quality_feedback:
            avg_satisfaction = np.mean([f['overall_satisfaction'] for f in quality_feedback])
            
            if avg_satisfaction < 3:
                suggestions.append({
                    'type': 'model_retraining',
                    'priority': 'high',
                    'description': f'Low satisfaction ({avg_satisfaction:.1f}/5) - Consider retraining with user feedback',
                    'action': 'Retrain models with positive/negative examples from user feedback'
                })
        
        # Analyze click-through rates by user
        users_with_low_ctr = []
        for user_id in set(f['user_id'] for f in self.feedback_collector.feedback_data):
            preferences = self.feedback_collector.analyze_user_preferences(user_id)
            if preferences.get('click_through_rate', 0) < 0.1:
                users_with_low_ctr.append(user_id)
        
        if len(users_with_low_ctr) > 0:
            suggestions.append({
                'type': 'personalization',
                'priority': 'medium',
                'description': f'{len(users_with_low_ctr)} users have low engagement',
                'action': 'Implement user-specific preference weighting'
            })
        
        return suggestions


def demo_feedback_system():
    """Demonstrate the feedback system with simulated user interactions."""
    print("🔄 Initializing User Feedback System Demo")
    
    # Initialize feedback collector
    collector = UserFeedbackCollector()
    
    # Simulate some user interactions
    print("\n📝 Simulating user interactions...")
    
    # User 1: Researcher interested in housing data
    collector.record_search_interaction(
        user_id="researcher_001",
        session_id="session_123",
        query="housing data singapore",
        results=[
            {'title': 'HDB Resale Prices', 'score': 0.95},
            {'title': 'Property Market Trends', 'score': 0.87},
            {'title': 'Housing Supply Data', 'score': 0.82}
        ],
        interaction_type="click",
        dataset_id="hdb_001",
        rating=5
    )
    
    # User 1: Downloads dataset
    collector.record_search_interaction(
        user_id="researcher_001",
        session_id="session_123", 
        query="housing data singapore",
        results=[],
        interaction_type="download",
        dataset_id="hdb_001"
    )
    
    # User 2: Student looking for transport data
    collector.record_search_interaction(
        user_id="student_002",
        session_id="session_456",
        query="transport statistics",
        results=[
            {'title': 'LTA Traffic Data', 'score': 0.91},
            {'title': 'MRT Ridership', 'score': 0.85},
            {'title': 'Bus Route Information', 'score': 0.78}
        ],
        interaction_type="click",
        dataset_id="lta_001",
        rating=4
    )
    
    # Record query expansion feedback
    collector.record_query_expansion_feedback(
        user_id="researcher_001",
        session_id="session_123",
        original_query="housing data",
        expanded_query="housing data hdb resale flat property market",
        expansion_helpful=True,
        preferred_terms=["hdb", "resale flat"]
    )
    
    # Record recommendation quality feedback
    collector.record_recommendation_quality(
        user_id="student_002",
        session_id="session_456",
        query="transport statistics",
        recommendations=[
            {'title': 'LTA Traffic Data'},
            {'title': 'MRT Ridership'},
            {'title': 'Bus Route Information'}
        ],
        overall_satisfaction=4,
        relevance_ratings=[5, 4, 3]
    )
    
    # Analyze feedback
    print("\n📊 Analyzing feedback data...")
    
    # User preferences
    user1_prefs = collector.analyze_user_preferences("researcher_001")
    print(f"\nUser 1 Preferences: {json.dumps(user1_prefs, indent=2)}")
    
    # Feedback summary
    summary = collector.get_feedback_summary()
    print(f"\nFeedback Summary: {json.dumps(summary, indent=2)}")
    
    # Model improvement suggestions
    improver = FeedbackDrivenModelImprover(collector)
    suggestions = improver.suggest_model_improvements()
    print(f"\nImprovement Suggestions: {json.dumps(suggestions, indent=2)}")
    
    # Save feedback
    collector.save_feedback()
    print("\n✅ Feedback system demo complete!")


if __name__ == "__main__":
    demo_feedback_system()