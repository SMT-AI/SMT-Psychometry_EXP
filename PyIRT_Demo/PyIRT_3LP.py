#!/usr/bin/env python3
"""
Robust 3-Parameter Logistic (3PL) IRT Psychometric Testing System
=================================================================

A comprehensive educational assessment system implementing:
- 3PL IRT model (difficulty, discrimination, guessing parameters)
- Robust error handling for insufficient data scenarios
- Classical Test Theory fallback
- Enhanced reliability calculations

Requirements:
pip install py-irt torch numpy pandas matplotlib seaborn scikit-learn

Author: Advanced Psychometric Assessment System
Version: 3.0 (3PL Implementation)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import multiple IRT libraries for robust analysis
try:
    from py_irt.models.one_param_logistic import OneParamLog
    from py_irt.models.two_param_logistic import TwoParamLog
    PYIRT_AVAILABLE = True
except ImportError:
    PYIRT_AVAILABLE = False

try:
    from girth import twopl_mml, onepl_mml, threepl_mml
    GIRTH_AVAILABLE = True
    print("✓ girth library available for IRT analysis")
except ImportError:
    GIRTH_AVAILABLE = False

# Fallback to the older pyirt library if available
try:
    from pyirt import irt as pyirt_irt
    PYIRT_OLD_AVAILABLE = True
    print("✓ pyirt (old) library available")
except ImportError:
    PYIRT_OLD_AVAILABLE = False

if not (PYIRT_AVAILABLE or GIRTH_AVAILABLE or PYIRT_OLD_AVAILABLE):
    print("No IRT libraries found. Install with:")
    print("  pip install girth  # Recommended")
    print("  pip install py-irt  # Alternative")
    print("  pip install pyirt   # Older alternative")

# Check available libraries
IRT_LIBRARIES = []
if GIRTH_AVAILABLE:
    IRT_LIBRARIES.append("girth")
if PYIRT_AVAILABLE:
    IRT_LIBRARIES.append("py-irt")
if PYIRT_OLD_AVAILABLE:
    IRT_LIBRARIES.append("pyirt")

print(f"Available IRT libraries: {IRT_LIBRARIES}")

class PsychometricItem:
    """Enhanced item class with 3PL parameters"""
    
    def __init__(self, item_id: str, question: str, options: List[str], 
                 correct_answer: int, difficulty: float = 0.0, 
                 discrimination: float = 1.0, guessing: float = 0.25, 
                 subject_area: str = "General"):
        self.item_id = item_id
        self.question = question
        self.options = options
        self.correct_answer = correct_answer  # 0-based index
        
        # 3PL IRT Parameters
        self.difficulty = difficulty  # b parameter (item difficulty)
        self.discrimination = discrimination  # a parameter (item discrimination)
        self.guessing = guessing  # c parameter (lower asymptote/guessing)
        
        self.subject_area = subject_area
        self.responses = []  # Store all responses to this item
        
        # Classical Test Theory statistics
        self.p_value = None  # Proportion correct (difficulty in CTT)
        self.point_biserial = None  # Item-total correlation
        self.response_frequencies = {i: 0 for i in range(len(options))}
        
    def display_question(self) -> str:
        """Format question for display"""
        display = f"\n{self.question}\n"
        for i, option in enumerate(self.options):
            display += f"{chr(65+i)}. {option}\n"
        return display
    
    def is_correct(self, response: int) -> bool:
        """Check if response is correct"""
        return response == self.correct_answer
    
    def add_response(self, student_id: str, response: int, is_correct: bool):
        """Record a student response"""
        self.responses.append({
            'student_id': student_id,
            'response': response,
            'is_correct': is_correct,
            'timestamp': datetime.now()
        })
        
        # Update response frequencies
        if 0 <= response < len(self.options):
            self.response_frequencies[response] += 1
    
    def calculate_classical_statistics(self, total_scores: Dict[str, int]):
        """Calculate classical test theory item statistics"""
        if len(self.responses) < 2:
            self.p_value = 0.0
            self.point_biserial = 0.0
            return
        
        # Calculate p-value (proportion correct)
        correct_responses = sum(1 for r in self.responses if r['is_correct'])
        self.p_value = correct_responses / len(self.responses)
        
        # Calculate point-biserial correlation
        try:
            item_scores = []
            student_totals = []
            
            for response in self.responses:
                student_id = response['student_id']
                item_scores.append(1 if response['is_correct'] else 0)
                student_totals.append(total_scores.get(student_id, 0))
            
            if len(set(item_scores)) > 1 and len(set(student_totals)) > 1:
                correlation_matrix = np.corrcoef(item_scores, student_totals)
                self.point_biserial = correlation_matrix[0, 1]
            else:
                self.point_biserial = 0.0
                
        except (ValueError, IndexError):
            self.point_biserial = 0.0
    
    def get_3pl_probability(self, ability: float) -> float:
        """Calculate probability of correct response using 3PL model"""
        # 3PL Formula: P(θ) = c + (1-c) * [1 / (1 + exp(-1.7 * a * (θ - b)))]
        try:
            exponent = -1.7 * self.discrimination * (ability - self.difficulty)
            two_pl_prob = 1 / (1 + np.exp(exponent))
            three_pl_prob = self.guessing + (1 - self.guessing) * two_pl_prob
            return max(0.001, min(0.999, three_pl_prob))  # Bound probabilities
        except (OverflowError, ZeroDivisionError):
            return self.guessing + 0.1  # Safe fallback

class Student:
    """Enhanced student class with detailed tracking"""
    
    def __init__(self, student_id: str, name: str, grade_level: str):
        self.student_id = student_id
        self.name = name
        self.grade_level = grade_level
        self.responses = {}  # item_id -> response
        
        # IRT parameters
        self.ability_estimate = 0.0  # θ (theta) parameter
        self.ability_se = 1.0  # Standard error of ability estimate
        
        # Test session information
        self.test_start_time = None
        self.test_end_time = None
        self.total_time_minutes = 0.0
        
        # Performance metrics
        self.raw_score = 0
        self.percent_correct = 0.0
        self.items_attempted = 0
        
    def record_response(self, item_id: str, response: int):
        """Record student's response to an item"""
        self.responses[item_id] = response
        
    def calculate_performance_metrics(self, items: Dict[str, PsychometricItem]):
        """Calculate various performance metrics"""
        if not self.responses:
            return
            
        self.items_attempted = len(self.responses)
        self.raw_score = sum(1 for item_id, response in self.responses.items()
                           if item_id in items and items[item_id].is_correct(response))
        
        self.percent_correct = (self.raw_score / self.items_attempted * 100) if self.items_attempted > 0 else 0.0
        
        # Calculate test duration
        if self.test_start_time and self.test_end_time:
            duration = self.test_end_time - self.test_start_time
            self.total_time_minutes = duration.total_seconds() / 60.0

class RobustIRTAnalyzer:
    """Robust IRT analysis with comprehensive error handling"""
    
    def __init__(self, min_responses: int = 10, min_students: int = 5, min_items: int = 3):
        self.min_responses = min_responses
        self.min_students = min_students
        self.min_items = min_items
        self.supported_models = self._check_supported_models()
        
    def _check_supported_models(self) -> List[str]:
        """Check which IRT models are supported by available libraries"""
        supported = []
        
        if GIRTH_AVAILABLE:
            supported.extend(['1PL_girth', '2PL_girth', '3PL_girth'])
            print("✓ girth library: 1PL, 2PL, 3PL models available")
        
        if PYIRT_OLD_AVAILABLE:
            supported.extend(['1PL_pyirt', '2PL_pyirt', '3PL_pyirt'])
            print("✓ pyirt library: 1PL, 2PL, 3PL models available")
        
        if PYIRT_AVAILABLE:
            try:
                OneParamLog(priors='vague', num_items=2, num_subjects=2, verbose=False)
                supported.append('1PL_py-irt')
            except Exception as e:
                print(f"py-irt 1PL model not available: {e}")
            
            try:
                TwoParamLog(priors='vague', num_items=2, num_subjects=2, verbose=False)
                supported.append('2PL_py-irt')
            except Exception as e:
                print(f"py-irt 2PL model not available: {e}")
        
        print(f"Supported IRT models: {supported}")
        return supported
        
    def validate_data_sufficiency(self, response_matrix: np.ndarray, 
                                student_ids: List[str], item_ids: List[str]) -> Tuple[bool, str]:
        """Check if data is sufficient for IRT analysis"""
        
        # Check basic dimensions
        n_students, n_items = response_matrix.shape
        
        if n_students < self.min_students:
            return False, f"Insufficient students: {n_students} < {self.min_students} required"
        
        if n_items < self.min_items:
            return False, f"Insufficient items: {n_items} < {self.min_items} required"
        
        # Check for valid responses
        valid_responses = np.sum(response_matrix != -1)
        if valid_responses < self.min_responses:
            return False, f"Insufficient responses: {valid_responses} < {self.min_responses} required"
        
        # Check for variance in responses
        valid_matrix = response_matrix[response_matrix != -1]
        if len(np.unique(valid_matrix)) < 2:
            return False, "No variance in responses (all correct or all incorrect)"
        
        # Check each item has some responses
        item_response_counts = np.sum(response_matrix != -1, axis=0)
        items_with_no_responses = np.sum(item_response_counts == 0)
        if items_with_no_responses > 0:
            return False, f"{items_with_no_responses} items have no responses"
        
        # Check each student has some responses
        student_response_counts = np.sum(response_matrix != -1, axis=1)
        students_with_no_responses = np.sum(student_response_counts == 0)
        if students_with_no_responses > 0:
            return False, f"{students_with_no_responses} students have no responses"
        
        return True, "Data sufficient for IRT analysis"
    
    def clean_response_matrix(self, response_matrix: np.ndarray, 
                            student_ids: List[str], item_ids: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Clean response matrix by removing problematic rows/columns"""
        
        original_students = len(student_ids)
        original_items = len(item_ids)
        
        # Remove students with no responses
        student_mask = np.any(response_matrix != -1, axis=1)
        response_matrix = response_matrix[student_mask]
        student_ids = [sid for i, sid in enumerate(student_ids) if student_mask[i]]
        
        # Remove items with no responses
        item_mask = np.any(response_matrix != -1, axis=0)
        response_matrix = response_matrix[:, item_mask]
        item_ids = [iid for i, iid in enumerate(item_ids) if item_mask[i]]
        
        # Remove items with no variance (all correct or all incorrect)
        item_variances = []
        for j in range(response_matrix.shape[1]):
            item_responses = response_matrix[:, j]
            valid_responses = item_responses[item_responses != -1]
            if len(valid_responses) > 1:
                variance = np.var(valid_responses)
                item_variances.append(variance > 0.001)  # Small threshold for numerical stability
            else:
                item_variances.append(False)
        
        variance_mask = np.array(item_variances)
        response_matrix = response_matrix[:, variance_mask]
        item_ids = [iid for i, iid in enumerate(item_ids) if variance_mask[i]]
        
        removed_students = original_students - len(student_ids)
        removed_items = original_items - len(item_ids)
        
        if removed_students > 0 or removed_items > 0:
            print(f"Data cleaning: Removed {removed_students} students and {removed_items} items")
        
        return response_matrix, student_ids, item_ids
    
    def run_3pl_analysis(self, response_matrix: np.ndarray, 
                        student_ids: List[str], item_ids: List[str]) -> Tuple[bool, Dict, Dict, str]:
        """Run IRT analysis using available IRT libraries with robust error handling"""
        
        # Validate and clean data
        response_matrix, student_ids, item_ids = self.clean_response_matrix(
            response_matrix, student_ids, item_ids)
        
        is_sufficient, message = self.validate_data_sufficiency(
            response_matrix, student_ids, item_ids)
        
        if not is_sufficient:
            return False, {}, {}, f"IRT Analysis failed: {message}"
        
        if not any([GIRTH_AVAILABLE, PYIRT_OLD_AVAILABLE, PYIRT_AVAILABLE]):
            return False, {}, {}, "No IRT libraries available"
        
        try:
            print(f"Running IRT analysis on {np.sum(response_matrix != -1)} responses...")
            print(f"Students: {len(student_ids)}, Items: {len(item_ids)}")
            print(f"Available libraries: {IRT_LIBRARIES}")
            
            # Create binary response matrix for IRT libraries
            # Replace missing values (-1) with NaN or handle appropriately
            irt_matrix = np.where(response_matrix == -1, np.nan, response_matrix).astype(float)
            
            # Try different IRT implementations in order of preference
            
            # 1. Try girth library (most reliable and complete)
            if GIRTH_AVAILABLE:
                try:
                    return self._run_girth_analysis(irt_matrix, student_ids, item_ids)
                except Exception as e:
                    print(f"Girth IRT analysis failed: {e}")
            
            # 2. Try old pyirt library
            if PYIRT_OLD_AVAILABLE:
                try:
                    return self._run_pyirt_old_analysis(irt_matrix, student_ids, item_ids)
                except Exception as e:
                    print(f"PyIRT (old) analysis failed: {e}")
            
            # 3. Try py-irt library (last resort)
            if PYIRT_AVAILABLE:
                try:
                    return self._run_pyirt_new_analysis(irt_matrix, student_ids, item_ids)
                except Exception as e:
                    print(f"py-irt analysis failed: {e}")
            
            return False, {}, {}, "All IRT libraries failed"
            
        except Exception as e:
            error_msg = f"IRT analysis failed: {str(e)}"
            print(error_msg)
            return False, {}, {}, error_msg
    
    def _run_girth_analysis(self, irt_matrix: np.ndarray, 
                           student_ids: List[str], item_ids: List[str]) -> Tuple[bool, Dict, Dict, str]:
        """Run IRT analysis using the girth library"""
        print("Attempting girth library analysis...")
        
        # Try 3PL model first
        try:
            print("  Trying 3PL model...")
            estimates = threepl_mml(irt_matrix)
            
            item_params = self._format_girth_item_params(estimates, item_ids, '3PL')
            user_params = self._format_girth_user_params(estimates, student_ids)
            
            return True, item_params, user_params, "3PL IRT analysis completed successfully (girth)"
            
        except Exception as e:
            print(f"  3PL failed: {e}")
        
        # Try 2PL model
        try:
            print("  Trying 2PL model...")
            estimates = twopl_mml(irt_matrix)
            
            item_params = self._format_girth_item_params(estimates, item_ids, '2PL')
            user_params = self._format_girth_user_params(estimates, student_ids)
            
            return True, item_params, user_params, "2PL IRT analysis completed successfully (girth)"
            
        except Exception as e:
            print(f"  2PL failed: {e}")
        
        # Try 1PL/Rasch model
        try:
            print("  Trying 1PL (Rasch) model...")
            estimates = onepl_mml(irt_matrix)
            
            item_params = self._format_girth_item_params(estimates, item_ids, '1PL')
            user_params = self._format_girth_user_params(estimates, student_ids)
            
            return True, item_params, user_params, "1PL (Rasch) IRT analysis completed successfully (girth)"
            
        except Exception as e:
            print(f"  1PL failed: {e}")
            raise Exception("All girth models failed")
    
    def _run_pyirt_old_analysis(self, irt_matrix: np.ndarray,
                               student_ids: List[str], item_ids: List[str]) -> Tuple[bool, Dict, Dict, str]:
        """Run IRT analysis using the old pyirt library"""
        print("Attempting pyirt (old) library analysis...")
        
        # Convert matrix to pyirt format (list of tuples)
        data_tuples = []
        for i, student_id in enumerate(student_ids):
            for j, item_id in enumerate(item_ids):
                if not np.isnan(irt_matrix[i, j]):
                    data_tuples.append((student_id, item_id, int(irt_matrix[i, j])))
        
        try:
            print("  Trying 2PL model...")
            item_param, user_param = pyirt_irt(data_tuples, 
                                              theta_bnds=[-4, 4], 
                                              alpha_bnds=[0.1, 3], 
                                              beta_bnds=[-4, 4])
            
            # Add guessing parameter for 3PL compatibility
            for item_id in item_param:
                item_param[item_id]['c'] = 0.25
            
            return True, item_param, user_param, "2PL IRT analysis completed successfully (pyirt)"
            
        except Exception as e:
            print(f"  PyIRT 2PL failed: {e}")
            
            try:
                print("  Trying 1PL model...")
                item_param, user_param = pyirt_irt(data_tuples, 
                                                  theta_bnds=[-4, 4], 
                                                  beta_bnds=[-4, 4])
                
                # Add default parameters for compatibility
                for item_id in item_param:
                    if 'alpha' not in item_param[item_id]:
                        item_param[item_id]['alpha'] = 1.0
                    item_param[item_id]['c'] = 0.25
                
                return True, item_param, user_param, "1PL IRT analysis completed successfully (pyirt)"
                
            except Exception as e2:
                print(f"  PyIRT 1PL failed: {e2}")
                raise Exception("All pyirt models failed")
    
    def _run_pyirt_new_analysis(self, irt_matrix: np.ndarray,
                               student_ids: List[str], item_ids: List[str]) -> Tuple[bool, Dict, Dict, str]:
        """Run IRT analysis using the new py-irt library (placeholder)"""
        print("py-irt library analysis not fully implemented (API unclear)")
        raise Exception("py-irt library integration incomplete")
    
    def _format_girth_item_params(self, estimates: Dict, item_ids: List[str], model_type: str) -> Dict:
        """Format girth estimation results to our standard format"""
        item_params = {}
        
        # Get parameter arrays from girth results
        discrimination = estimates.get('Discrimination', np.ones(len(item_ids)))
        difficulty = estimates.get('Difficulty', np.zeros(len(item_ids)))
        guessing = estimates.get('Guessing', np.full(len(item_ids), 0.25))
        
        for i, item_id in enumerate(item_ids):
            item_params[item_id] = {
                'alpha': float(discrimination[i]) if i < len(discrimination) else 1.0,
                'beta': float(difficulty[i]) if i < len(difficulty) else 0.0,
                'c': float(guessing[i]) if i < len(guessing) else 0.25
            }
        
        return item_params
    
    def _format_girth_user_params(self, estimates: Dict, student_ids: List[str]) -> Dict:
        """Format girth user ability estimates to our standard format"""
        user_params = {}
        
        # Get ability estimates (theta values)
        abilities = estimates.get('Ability', np.zeros(len(student_ids)))
        
        for i, student_id in enumerate(student_ids):
            user_params[student_id] = {
                'theta': float(abilities[i]) if i < len(abilities) else 0.0,
                'se': 1.0  # Standard error not typically provided by girth
            }
        
        return user_params

class EnhancedPsychometricTest:
    """Enhanced test management with robust 3PL IRT implementation"""
    
    def __init__(self, test_name: str, test_id: str):
        self.test_name = test_name
        self.test_id = test_id
        self.items = {}  # item_id -> PsychometricItem
        self.students = {}  # student_id -> Student
        
        # IRT Analysis components
        self.irt_analyzer = RobustIRTAnalyzer()
        self.item_parameters = {}  # IRT parameters
        self.ability_estimates = {}  # Student ability estimates
        self.irt_analysis_successful = False
        self.irt_analysis_message = ""
        
        # Test statistics
        self.reliability_alpha = 0.0
        self.test_statistics = {}
        
    def add_item(self, item: PsychometricItem):
        """Add an item to the test"""
        self.items[item.item_id] = item
        
    def register_student(self, student: Student):
        """Register a student for the test"""
        self.students[student.student_id] = student
        
    def administer_test(self, student_id: str) -> bool:
        """Administer test to a specific student via terminal"""
        if student_id not in self.students:
            print(f"Student {student_id} not registered!")
            return False
            
        student = self.students[student_id]
        student.test_start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"PSYCHOMETRIC ASSESSMENT: {self.test_name}")
        print(f"Student: {student.name} (ID: {student.student_id})")
        print(f"Grade Level: {student.grade_level}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Answer each question by typing the letter (A, B, C, or D)")
        print("- Type 'skip' to skip a question")
        print("- Type 'quit' to end the test early")
        print(f"- Total Questions: {len(self.items)}")
        print("\nPress Enter to begin...")
        input()
        
        question_num = 1
        for item_id, item in self.items.items():
            print(f"\n{'='*40}")
            print(f"Question {question_num} of {len(self.items)}")
            print(f"Subject: {item.subject_area}")
            print(item.display_question())
            
            while True:
                response = input("Your answer (A/B/C/D, 'skip', or 'quit'): ").strip().upper()
                
                if response == 'QUIT':
                    print("Test ended early by student choice.")
                    student.test_end_time = datetime.now()
                    student.calculate_performance_metrics(self.items)
                    return False
                elif response == 'SKIP':
                    print("Question skipped.")
                    break
                elif response in ['A', 'B', 'C', 'D']:
                    response_idx = ord(response) - ord('A')
                    student.record_response(item_id, response_idx)
                    item.add_response(student_id, response_idx, 
                                    item.is_correct(response_idx))
                    break
                else:
                    print("Invalid input. Please enter A, B, C, D, 'skip', or 'quit'.")
            
            question_num += 1
            
        student.test_end_time = datetime.now()
        student.calculate_performance_metrics(self.items)
        
        print(f"\n{'='*60}")
        print("Test completed! Thank you for your participation.")
        print(f"Raw Score: {student.raw_score}/{len(self.items)} ({student.percent_correct:.1f}%)")
        print(f"Time taken: {student.total_time_minutes:.1f} minutes")
        print(f"{'='*60}")
        
        return True
    
    def create_response_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create response matrix for IRT analysis"""
        completed_students = [sid for sid, student in self.students.items() 
                            if len(student.responses) > 0]
        
        if not completed_students:
            return np.array([]), [], []
        
        student_ids = completed_students
        item_ids = list(self.items.keys())
        
        if not item_ids:
            return np.array([]), student_ids, []
        
        # Create response matrix (students x items)
        matrix = np.full((len(student_ids), len(item_ids)), -1)  # -1 for missing responses
        
        for i, student_id in enumerate(student_ids):
            student = self.students[student_id]
            for j, item_id in enumerate(item_ids):
                if item_id in student.responses:
                    response = student.responses[item_id]
                    is_correct = self.items[item_id].is_correct(response)
                    matrix[i, j] = 1 if is_correct else 0
                    
        return matrix, student_ids, item_ids
    
    def calculate_classical_statistics(self):
        """Calculate classical test theory statistics"""
        completed_students = {sid: student for sid, student in self.students.items() 
                            if len(student.responses) > 0}
        
        if len(completed_students) < 2:
            self.reliability_alpha = 0.0
            return
        
        # Calculate total scores for point-biserial correlations
        total_scores = {}
        for student_id, student in completed_students.items():
            total_scores[student_id] = student.raw_score
        
        # Calculate item statistics
        for item in self.items.values():
            item.calculate_classical_statistics(total_scores)
        
        # Calculate Cronbach's alpha
        self.reliability_alpha = self._calculate_cronbach_alpha()
    
    def _calculate_cronbach_alpha(self) -> float:
        """Calculate Cronbach's alpha reliability coefficient"""
        response_matrix, student_ids, item_ids = self.create_response_matrix()
        
        if response_matrix.size == 0 or len(student_ids) < 2 or len(item_ids) < 2:
            return 0.0
        
        try:
            # Create matrix with only valid responses
            valid_data = []
            for i in range(response_matrix.shape[0]):
                row = response_matrix[i, :]
                if np.all(row != -1):  # Only complete responses
                    valid_data.append(row)
            
            if len(valid_data) < 2:
                return 0.0
            
            valid_matrix = np.array(valid_data)
            n_items = valid_matrix.shape[1]
            
            if n_items < 2:
                return 0.0
            
            # Calculate variances
            item_variances = np.var(valid_matrix, axis=0, ddof=1)
            total_scores = np.sum(valid_matrix, axis=1)
            total_variance = np.var(total_scores, ddof=1)
            
            if total_variance == 0:
                return 0.0
            
            # Cronbach's alpha formula
            alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
            return max(0.0, min(1.0, alpha))  # Bound between 0 and 1
            
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            return 0.0
    
    def run_irt_analysis(self):
        """Run comprehensive IRT analysis with robust error handling"""
        print("\n" + "="*60)
        print("RUNNING 3PL IRT ANALYSIS")
        print("="*60)
        
        # First calculate classical statistics
        self.calculate_classical_statistics()
        
        response_matrix, student_ids, item_ids = self.create_response_matrix()
        
        if response_matrix.size == 0:
            self.irt_analysis_successful = False
            self.irt_analysis_message = "No response data available"
            print("Warning: No response data available for analysis")
            return False
        
        # Run 3PL IRT analysis
        success, item_params, user_params, message = self.irt_analyzer.run_3pl_analysis(
            response_matrix, student_ids, item_ids)
        
        self.irt_analysis_successful = success
        self.irt_analysis_message = message
        
        if success:
            self.item_parameters = item_params
            self.ability_estimates = user_params
            
            # Update item parameters
            for item_id, params in item_params.items():
                if item_id in self.items:
                    self.items[item_id].difficulty = params.get('beta', 0.0)
                    self.items[item_id].discrimination = params.get('alpha', 1.0)
                    self.items[item_id].guessing = params.get('c', 0.25)
            
            # Update student ability estimates
            for student_id, params in user_params.items():
                if student_id in self.students:
                    self.students[student_id].ability_estimate = params.get('theta', 0.0)
                    self.students[student_id].ability_se = params.get('se', 1.0)
            
            print(f"✓ {message}")
            print(f"  - Items analyzed: {len(item_params)}")
            print(f"  - Students analyzed: {len(user_params)}")
        else:
            print(f"✗ {message}")
            print("  Falling back to Classical Test Theory analysis")
        
        return success
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report with robust error handling"""
        report = []
        report.append("="*80)
        report.append(f"PSYCHOMETRIC TEST REPORT: {self.test_name}")
        report.append("="*80)
        report.append(f"Test ID: {self.test_id}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Test Overview
        report.append("TEST OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Items: {len(self.items)}")
        report.append(f"Total Students Registered: {len(self.students)}")
        
        # Count completed tests
        completed_students = [s for s in self.students.values() if len(s.responses) > 0]
        completed_count = len(completed_students)
        report.append(f"Completed Tests: {completed_count}")
        
        if completed_count == 0:
            report.append("\nNo completed tests available for analysis.")
            report.append("="*80)
            return "\n".join(report)
        
        # IRT Analysis Status
        report.append("")
        report.append("ANALYSIS METHOD")
        report.append("-" * 40)
        if self.irt_analysis_successful:
            report.append("✓ 3-Parameter Logistic (3PL) IRT Analysis")
        else:
            report.append("✗ 3PL IRT Analysis Failed")
            report.append(f"  Reason: {self.irt_analysis_message}")
            report.append("✓ Classical Test Theory Analysis (Fallback)")
        report.append("")
        
        # Student Performance Summary
        report.append("STUDENT PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        raw_scores = [s.raw_score for s in completed_students]
        percent_scores = [s.percent_correct for s in completed_students]
        
        if raw_scores:
            max_score = len(self.items)
            
            report.append(f"Mean Score: {np.mean(raw_scores):.2f} / {max_score} ({np.mean(percent_scores):.1f}%)")
            report.append(f"Standard Deviation: {np.std(raw_scores):.2f}")
            report.append(f"Range: {np.min(raw_scores)} - {np.max(raw_scores)}")
            report.append(f"Reliability (Cronbach's α): {self.reliability_alpha:.3f}")
            
            # Reliability interpretation
            if self.reliability_alpha >= 0.9:
                reliability_desc = "Excellent"
            elif self.reliability_alpha >= 0.8:
                reliability_desc = "Good"
            elif self.reliability_alpha >= 0.7:
                reliability_desc = "Acceptable"
            elif self.reliability_alpha >= 0.6:
                reliability_desc = "Questionable"
            else:
                reliability_desc = "Poor"
            
            report.append(f"Reliability Assessment: {reliability_desc}")
            report.append("")
            
            # Individual student results
            report.append("INDIVIDUAL STUDENT RESULTS")
            report.append("-" * 40)
            
            header = f"{'Student ID':<12} {'Name':<20} {'Score':<8} {'Percent':<8}"
            if self.irt_analysis_successful:
                header += f" {'Ability (θ)':<12} {'SE':<8}"
            header += f" {'Grade':<8} {'Time (min)'}"
            
            report.append(header)
            report.append("-" * len(header))
            
            for student in completed_students:
                line = (f"{student.student_id:<12} {student.name:<20} "
                       f"{student.raw_score}/{max_score:<6} {student.percent_correct:<7.1f}%")
                
                if self.irt_analysis_successful:
                    line += f" {student.ability_estimate:<11.3f} {student.ability_se:<7.3f}"
                
                line += f" {student.grade_level:<8} {student.total_time_minutes:<7.1f}"
                report.append(line)
            report.append("")
        
        # Item Analysis
        report.append("ITEM ANALYSIS")
        report.append("-" * 40)
        
        if self.irt_analysis_successful:
            header = (f"{'Item ID':<10} {'Subject':<15} {'Difficulty':<12} "
                     f"{'Discrim.':<10} {'Guessing':<10} {'P-Value':<10} {'Responses'}")
        else:
            header = (f"{'Item ID':<10} {'Subject':<15} {'P-Value':<10} "
                     f"{'Point-Bis.':<12} {'Responses'}")
        
        report.append(header)
        report.append("-" * len(header))
        
        for item_id, item in self.items.items():
            responses_count = len(item.responses)
            
            if self.irt_analysis_successful:
                line = (f"{item_id:<10} {item.subject_area:<15} "
                       f"{item.difficulty:<12.3f} {item.discrimination:<10.3f} "
                       f"{item.guessing:<10.3f} {item.p_value or 0:<10.3f} {responses_count}")
            else:
                point_bis = item.point_biserial if item.point_biserial is not None else 0.0
                p_val = item.p_value if item.p_value is not None else 0.0
                line = (f"{item_id:<10} {item.subject_area:<15} "
                       f"{p_val:<10.3f} {point_bis:<12.3f} {responses_count}")
            
            report.append(line)
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if completed_count < 30:
            report.append("• Collect more responses (target: 100+) for reliable IRT analysis")
        
        if self.reliability_alpha < 0.7:
            report.append("• Review item quality - low reliability may indicate measurement issues")
        
        # Item-specific recommendations
        problem_items = []
        for item in self.items.values():
            if item.p_value is not None:
                if item.p_value < 0.2:
                    problem_items.append(f"{item.item_id} (too difficult)")
                elif item.p_value > 0.9:
                    problem_items.append(f"{item.item_id} (too easy)")
            
            if item.point_biserial is not None and item.point_biserial < 0.2:
                problem_items.append(f"{item.item_id} (poor discrimination)")
        
        if problem_items:
            report.append("• Review these items: " + ", ".join(problem_items))
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save test results to files with robust error handling"""
        if filename is None:
            filename = f"{self.test_id}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Save detailed report
            report = self.generate_test_report()
            with open(f"{filename}_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save raw data as CSV only if there are completed tests
            completed_students = [s for s in self.students.values() if len(s.responses) > 0]
            
            if completed_students:
                response_data = []
                for student_id, student in self.students.items():
                    if len(student.responses) > 0:
                        for item_id, response in student.responses.items():
                            if item_id in self.items:
                                item = self.items[item_id]
                                response_data.append({
                                    'student_id': student_id,
                                    'student_name': student.name,
                                    'grade_level': student.grade_level,
                                    'item_id': item_id,
                                    'subject_area': item.subject_area,
                                    'response': response,
                                    'response_letter': chr(65 + response) if 0 <= response < 4 else 'Invalid',
                                    'correct_answer': item.correct_answer,
                                    'correct_letter': chr(65 + item.correct_answer),
                                    'is_correct': item.is_correct(response),
                                    'raw_score': student.raw_score,
                                    'percent_correct': student.percent_correct,
                                    'ability_estimate': student.ability_estimate,
                                    'test_time_minutes': student.total_time_minutes
                                })
                
                if response_data:
                    df = pd.DataFrame(response_data)
                    df.to_csv(f"{filename}_raw_data.csv", index=False)
                    
                    # Save item statistics
                    item_stats = []
                    for item_id, item in self.items.items():
                        if len(item.responses) > 0:
                            item_stats.append({
                                'item_id': item_id,
                                'subject_area': item.subject_area,
                                'question': item.question[:100] + '...' if len(item.question) > 100 else item.question,
                                'correct_answer': chr(65 + item.correct_answer),
                                'difficulty_3pl': item.difficulty,
                                'discrimination_3pl': item.discrimination,
                                'guessing_3pl': item.guessing,
                                'p_value_classical': item.p_value or 0.0,
                                'point_biserial': item.point_biserial or 0.0,
                                'response_count': len(item.responses),
                                'percent_correct': (item.p_value * 100) if item.p_value else 0.0
                            })
                    
                    if item_stats:
                        item_df = pd.DataFrame(item_stats)
                        item_df.to_csv(f"{filename}_item_analysis.csv", index=False)
                    
                    # Save student summary
                    student_summary = []
                    for student in completed_students:
                        student_summary.append({
                            'student_id': student.student_id,
                            'name': student.name,
                            'grade_level': student.grade_level,
                            'raw_score': student.raw_score,
                            'max_possible': len(self.items),
                            'percent_correct': student.percent_correct,
                            'ability_estimate': student.ability_estimate,
                            'ability_se': student.ability_se,
                            'items_attempted': student.items_attempted,
                            'test_time_minutes': student.total_time_minutes,
                            'irt_analysis_used': self.irt_analysis_successful
                        })
                    
                    student_df = pd.DataFrame(student_summary)
                    student_df.to_csv(f"{filename}_student_summary.csv", index=False)
            
            # Save IRT parameters if available
            if self.irt_analysis_successful and self.item_parameters:
                irt_data = []
                for item_id, params in self.item_parameters.items():
                    irt_data.append({
                        'item_id': item_id,
                        'difficulty_b': params.get('beta', 0.0),
                        'discrimination_a': params.get('alpha', 1.0),
                        'guessing_c': params.get('c', 0.25),
                        'model': '3PL'
                    })
                
                if irt_data:
                    irt_df = pd.DataFrame(irt_data)
                    irt_df.to_csv(f"{filename}_irt_parameters.csv", index=False)
                
                # Save ability estimates
                ability_data = []
                for student_id, params in self.ability_estimates.items():
                    if student_id in self.students:
                        student = self.students[student_id]
                        ability_data.append({
                            'student_id': student_id,
                            'name': student.name,
                            'grade_level': student.grade_level,
                            'theta_ability': params.get('theta', 0.0),
                            'standard_error': params.get('se', 1.0),
                            'raw_score': student.raw_score
                        })
                
                if ability_data:
                    ability_df = pd.DataFrame(ability_data)
                    ability_df.to_csv(f"{filename}_ability_estimates.csv", index=False)
            
            print(f"\nResults saved successfully:")
            print(f"- {filename}_report.txt (Comprehensive report)")
            
            if completed_students:
                print(f"- {filename}_raw_data.csv (Individual responses)")
                print(f"- {filename}_item_analysis.csv (Item statistics)")
                print(f"- {filename}_student_summary.csv (Student performance)")
                
                if self.irt_analysis_successful:
                    print(f"- {filename}_irt_parameters.csv (3PL item parameters)")
                    print(f"- {filename}_ability_estimates.csv (Student abilities)")
            else:
                print("- No response data files saved (no completed tests)")
                
        except Exception as e:
            print(f"Error saving results: {e}")
            print("Attempting to save basic report only...")
            try:
                report = self.generate_test_report()
                with open(f"{filename}_report_basic.txt", 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Basic report saved as {filename}_report_basic.txt")
            except Exception as e2:
                print(f"Failed to save even basic report: {e2}")

def create_sample_test() -> EnhancedPsychometricTest:
    """Create a sample mathematics assessment with enhanced 3PL items"""
    test = EnhancedPsychometricTest("Advanced Middle School Mathematics Assessment", "MATH_3PL_001")
    
    # Enhanced math items with estimated 3PL parameters
    items = [
        PsychometricItem(
            "MATH_001", 
            "What is 15% of 80?",
            ["10", "12", "15", "20"],
            1,  # Correct answer: B (12)
            difficulty=-0.5,  # Easier item
            discrimination=1.2,
            guessing=0.25,
            subject_area="Percentages"
        ),
        PsychometricItem(
            "MATH_002",
            "If x + 7 = 15, what is the value of x?",
            ["6", "7", "8", "9"],
            2,  # Correct answer: C (8)
            difficulty=-0.2,
            discrimination=1.4,
            guessing=0.25,
            subject_area="Algebra"
        ),
        PsychometricItem(
            "MATH_003",
            "What is the area of a rectangle with length 8 cm and width 5 cm?",
            ["13 cm²", "26 cm²", "40 cm²", "45 cm²"],
            2,  # Correct answer: C (40 cm²)
            difficulty=0.1,
            discrimination=1.1,
            guessing=0.25,
            subject_area="Geometry"
        ),
        PsychometricItem(
            "MATH_004",
            "Which fraction is equivalent to 0.75?",
            ["1/4", "2/3", "3/4", "4/5"],
            2,  # Correct answer: C (3/4)
            difficulty=-0.3,
            discrimination=1.3,
            guessing=0.25,
            subject_area="Fractions"
        ),
        PsychometricItem(
            "MATH_005",
            "What is the mean of the numbers: 4, 6, 8, 10, 12?",
            ["6", "7", "8", "9"],
            2,  # Correct answer: C (8)
            difficulty=0.3,
            discrimination=1.0,
            guessing=0.25,
            subject_area="Statistics"
        ),
        PsychometricItem(
            "MATH_006",
            "If a triangle has angles of 60° and 70°, what is the third angle?",
            ["40°", "50°", "60°", "70°"],
            1,  # Correct answer: B (50°)
            difficulty=0.5,
            discrimination=1.2,
            guessing=0.25,
            subject_area="Geometry"
        ),
        PsychometricItem(
            "MATH_007",
            "What is 2³ × 2²?",
            ["16", "24", "32", "64"],
            2,  # Correct answer: C (32)
            difficulty=0.8,  # Harder item
            discrimination=1.5,
            guessing=0.25,
            subject_area="Exponents"
        ),
        PsychometricItem(
            "MATH_008",
            "Which number is a prime number?",
            ["15", "17", "21", "25"],
            1,  # Correct answer: B (17)
            difficulty=0.4,
            discrimination=1.1,
            guessing=0.25,
            subject_area="Number Theory"
        ),
        PsychometricItem(
            "MATH_009",
            "Solve for y: 3y - 12 = 21",
            ["9", "11", "12", "15"],
            1,  # Correct answer: B (11)
            difficulty=1.0,  # Challenging item
            discrimination=1.8,
            guessing=0.25,
            subject_area="Algebra"
        ),
        PsychometricItem(
            "MATH_010",
            "What is the circumference of a circle with radius 4 cm? (Use π ≈ 3.14)",
            ["12.56 cm", "25.12 cm", "50.24 cm", "78.54 cm"],
            1,  # Correct answer: B (25.12 cm)
            difficulty=0.7,
            discrimination=1.3,
            guessing=0.25,
            subject_area="Geometry"
        )
    ]
    
    # Add items to test
    for item in items:
        test.add_item(item)
    
    return test

def register_sample_students(test: EnhancedPsychometricTest) -> List[Student]:
    """Register sample students"""
    students = [
        Student("STU_001", "Alice Johnson", "Grade 7"),
        Student("STU_002", "Bob Smith", "Grade 7"),
        Student("STU_003", "Carol Davis", "Grade 8"),
        Student("STU_004", "David Wilson", "Grade 8"),
        Student("STU_005", "Emma Brown", "Grade 7"),
        Student("STU_006", "Frank Chen", "Grade 8"),
        Student("STU_007", "Grace Miller", "Grade 7"),
        Student("STU_008", "Henry Taylor", "Grade 8")
    ]
    
    for student in students:
        test.register_student(student)
    
    return students

def simulate_realistic_responses(test: EnhancedPsychometricTest, num_students: int = 50):
    """Simulate realistic student responses using 3PL model"""
    print(f"Simulating responses for {num_students} virtual students...")
    
    np.random.seed(42)  # For reproducible results
    
    # Create additional virtual students
    for i in range(len(test.students), num_students):
        grade = "Grade 7" if i % 2 == 0 else "Grade 8"
        student = Student(f"SIM_{i+1:03d}", f"Virtual Student {i+1}", grade)
        test.register_student(student)
    
    # Simulate responses for all students
    all_students = list(test.students.values())
    
    for student in all_students:
        if len(student.responses) == 0:  # Only simulate if not already completed
            # Assign random ability from normal distribution
            ability = np.random.normal(0, 1)
            student.ability_estimate = ability
            
            # Simulate test timing
            student.test_start_time = datetime.now()
            
            responses_correct = 0
            for item_id, item in test.items.items():
                # Use 3PL model to determine probability of correct response
                prob_correct = item.get_3pl_probability(ability)
                is_correct = np.random.random() < prob_correct
                
                if is_correct:
                    response = item.correct_answer
                    responses_correct += 1
                else:
                    # Select incorrect response (not purely random - some answers more attractive)
                    options = list(range(len(item.options)))
                    options.remove(item.correct_answer)
                    
                    # Add some bias toward certain incorrect answers
                    if len(options) >= 2:
                        weights = [0.4, 0.3, 0.3] if len(options) == 3 else [0.4, 0.3, 0.2, 0.1]
                        response = np.random.choice(options, p=weights[:len(options)])
                    else:
                        response = options[0]
                
                student.record_response(item_id, response)
                item.add_response(student.student_id, response, is_correct)
            
            # Simulate test end and calculate metrics
            test_duration = np.random.normal(25, 8)  # Average 25 minutes, SD 8
            test_duration = max(10, test_duration)  # Minimum 10 minutes
            
            student.test_end_time = student.test_start_time + pd.Timedelta(minutes=test_duration)
            student.calculate_performance_metrics(test.items)
    
    completed_count = sum(1 for s in test.students.values() if len(s.responses) > 0)
    print(f"✓ Simulation completed! {completed_count} students now have responses.")

def test_pyirt_compatibility():
    """Test IRT library compatibility and available models"""
    working_libraries = []
    
    # Test girth library
    if GIRTH_AVAILABLE:
        try:
            # Create minimal test data
            test_data = np.array([
                [1, 0, 1, 0],
                [0, 1, 1, 0], 
                [1, 1, 0, 1],
                [0, 0, 1, 1]
            ])
            
            # Test 2PL model
            try:
                estimates = twopl_mml(test_data)
                working_libraries.append("girth (2PL)")
                print("✓ girth 2PL model working")
            except Exception as e:
                print(f"✗ girth 2PL failed: {e}")
            
            # Test 1PL model
            try:
                estimates = onepl_mml(test_data)
                working_libraries.append("girth (1PL)")
                print("✓ girth 1PL model working")
            except Exception as e:
                print(f"✗ girth 1PL failed: {e}")
                
        except Exception as e:
            print(f"✗ girth library test failed: {e}")
    
    # Test old pyirt library
    if PYIRT_OLD_AVAILABLE:
        try:
            test_data_tuples = [
                ('student1', 'item1', 1),
                ('student1', 'item2', 0),
                ('student2', 'item1', 0),
                ('student2', 'item2', 1)
            ]
            
            # Test basic functionality
            try:
                item_param, user_param = pyirt_irt(test_data_tuples, theta_bnds=[-2, 2], beta_bnds=[-2, 2])
                working_libraries.append("pyirt (old)")
                print("✓ pyirt (old) basic model working")
            except Exception as e:
                print(f"✗ pyirt (old) failed: {e}")
                
        except Exception as e:
            print(f"✗ pyirt (old) library test failed: {e}")
    
    # Test py-irt library
    if PYIRT_AVAILABLE:
        try:
            model_1pl = OneParamLog(priors='vague', num_items=2, num_subjects=2, verbose=False)
            working_libraries.append("py-irt (1PL)")
            print("✓ py-irt 1PL model structure working")
            
            model_2pl = TwoParamLog(priors='vague', num_items=2, num_subjects=2, verbose=False)
            working_libraries.append("py-irt (2PL)")
            print("✓ py-irt 2PL model structure working")
            
        except Exception as e:
            print(f"✗ py-irt model test failed: {e}")
    
    if not working_libraries:
        print("✗ No IRT libraries are working")
        print("Install recommendations:")
        print("  pip install girth  # Most stable")
        print("  pip install pyirt  # Alternative") 
        print("  pip install py-irt  # Research library")
        return False
    
    print(f"✓ Working IRT libraries: {working_libraries}")
    return True

def main():
    """Enhanced main function with 3PL IRT capabilities"""
    print("="*80)
    print("ENHANCED PSYCHOMETRIC TESTING SYSTEM")
    print("3-Parameter Logistic (3PL) Item Response Theory")
    print("="*80)
    
    # Create enhanced test
    test = create_sample_test()
    students = register_sample_students(test)
    
    print(f"\nTest created: {test.test_name}")
    print(f"Items loaded: {len(test.items)} (with 3PL parameters)")
    print(f"Students registered: {len(test.students)}")
    
    # Main menu
    while True:
        print("\n" + "="*50)
        print("MAIN MENU - 3PL IRT SYSTEM")
        print("="*50)
        print("1. View Test Information")
        print("2. Administer Test to Student")
        print("3. Run 3PL IRT Analysis")
        print("4. Generate Test Report")
        print("5. Save Results")
        print("6. Simulate Student Responses (Demo)")
        print("7. Large-Scale Simulation (50+ students)")
        print("8. View Item Characteristic Curves")
        print("0. Exit")
        
        choice = input("\nSelect option (0-9): ").strip()
        
        if choice == "0":
            print("Thank you for using the 3PL IRT Testing System!")
            break
            
        elif choice == "1":
            print(f"\nTest: {test.test_name}")
            print(f"Model: 3-Parameter Logistic (3PL) IRT")
            print(f"Items: {len(test.items)}")
            print(f"Students: {len(test.students)}")
            
            completed_students = [s for s in test.students.values() if len(s.responses) > 0]
            print(f"Completed Tests: {len(completed_students)}")
            
            print(f"\nIRT Analysis Status: {'✓ Available' if len(completed_students) >= 5 else '✗ Need 5+ completed tests'}")
            print(f"Reliability Analysis: {'✓ Available' if len(completed_students) >= 2 else '✗ Need 2+ completed tests'}")
            
            print("\nRegistered Students (showing first 10):")
            for i, student in enumerate(list(test.students.values())[:10]):
                responses_count = len(student.responses)
                status = f"Completed ({responses_count} items)" if responses_count > 0 else "Not Started"
                print(f"  {student.student_id}: {student.name} ({student.grade_level}) - {status}")
            
            if len(test.students) > 10:
                print(f"  ... and {len(test.students) - 10} more students")
                
        elif choice == "2":
            real_students = [s for s in test.students.values() if not s.student_id.startswith("SIM_")]
            
            if not real_students:
                print("No real students available. All students are simulated.")
                continue
                
            print("\nAvailable Real Students:")
            for student in real_students:
                status = "Completed" if len(student.responses) > 0 else "Available"
                print(f"  {student.student_id}: {student.name} - {status}")
            
            student_id = input("\nEnter Student ID: ").strip()
            if student_id in test.students:
                if len(test.students[student_id].responses) > 0:
                    retry = input("Student has already taken the test. Retake? (y/N): ")
                    if retry.lower() != 'y':
                        continue
                    # Clear previous responses
                    student = test.students[student_id]
                    student.responses.clear()
                    # Remove student responses from items
                    for item in test.items.values():
                        item.responses = [r for r in item.responses if r['student_id'] != student_id]
                
                test.administer_test(student_id)
            else:
                print("Invalid Student ID!")
                
        elif choice == "3":
            completed_tests = sum(1 for s in test.students.values() if len(s.responses) > 0)
            if completed_tests < 5:
                print(f"Warning: Only {completed_tests} completed tests. Recommend 5+ for reliable 3PL analysis.")
                if completed_tests < 2:
                    print("Cannot perform any psychometric analysis with fewer than 2 completed tests.")
                    continue
                
                proceed = input("Proceed with limited data? (y/N): ")
                if proceed.lower() != 'y':
                    continue
            
            success = test.run_irt_analysis()
            
            if success:
                print("\n3PL IRT Analysis Results:")
                print(f"- Items analyzed: {len(test.item_parameters)}")
                print(f"- Students analyzed: {len(test.ability_estimates)}")
                print(f"- Reliability (α): {test.reliability_alpha:.3f}")
            
        elif choice == "4":
            print("\n" + test.generate_test_report())
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            filename = input("Enter filename prefix (or press Enter for default): ").strip()
            if not filename:
                filename = None
            test.save_results(filename)
            
        elif choice == "6":
            print("Simulating realistic responses for existing students...")
            
            # Simulate responses for non-completed students only
            simulate_count = 0
            for student in test.students.values():
                if len(student.responses) == 0 and not student.student_id.startswith("SIM_"):
                    simulate_count += 1
            
            if simulate_count == 0:
                print("All real students have already completed the test.")
                print("Use option 7 for large-scale simulation with virtual students.")
                continue
                
            # Use the existing simulation function but limit to registered students
            np.random.seed(42)
            for student in test.students.values():
                if len(student.responses) == 0:  # Only simulate if not already completed
                    ability = np.random.normal(0, 1)
                    student.ability_estimate = ability
                    student.test_start_time = datetime.now()
                    
                    for item_id, item in test.items.items():
                        prob_correct = item.get_3pl_probability(ability)
                        is_correct = np.random.random() < prob_correct
                        
                        if is_correct:
                            response = item.correct_answer
                        else:
                            options = list(range(len(item.options)))
                            options.remove(item.correct_answer)
                            response = np.random.choice(options)
                        
                        student.record_response(item_id, response)
                        item.add_response(student.student_id, response, is_correct)
                    
                    test_duration = np.random.normal(25, 8)
                    test_duration = max(10, test_duration)
                    student.test_end_time = student.test_start_time + pd.Timedelta(minutes=test_duration)
                    student.calculate_performance_metrics(test.items)
            
            completed_count = sum(1 for s in test.students.values() if len(s.responses) > 0)
            print(f"✓ Simulation completed! {completed_count} students now have responses.")
            
        elif choice == "7":
            num_students = input("Enter number of total students for simulation (default 50): ").strip()
            try:
                num_students = int(num_students) if num_students else 50
                num_students = max(10, min(200, num_students))  # Reasonable bounds
            except ValueError:
                num_students = 50
            
            simulate_realistic_responses(test, num_students)
            
        elif choice == "8":
            if not test.irt_analysis_successful:
                print("3PL IRT analysis must be completed first (option 3).")
                continue
                
            print("Generating Item Characteristic Curves...")
            try:
                # Simple text-based ICC display
                print("\nItem Characteristic Curves (3PL Model)")
                print("=" * 60)
                
                ability_range = np.linspace(-3, 3, 13)
                
                for item_id, item in list(test.items.items())[:3]:  # Show first 3 items
                    print(f"\nItem {item_id}: {item.subject_area}")
                    print(f"Difficulty: {item.difficulty:.3f}, Discrimination: {item.discrimination:.3f}, Guessing: {item.guessing:.3f}")
                    print("Ability  | Probability")
                    print("-" * 20)
                    
                    for ability in ability_range:
                        prob = item.get_3pl_probability(ability)
                        bar_length = int(prob * 20)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        print(f"{ability:6.1f}  | {prob:.3f} {bar}")
                
                print(f"\nShowing first 3 items. Total items analyzed: {len(test.items)}")
                
            except Exception as e:
                print(f"Error generating curves: {e}")
            
            input("\nPress Enter to continue...")
            
        elif choice == "9":
            print("\nTesting py-irt compatibility...")
            working = test_pyirt_compatibility()
            
            if working:
                print("✓ py-irt is working correctly")
            else:
                print("✗ py-irt has compatibility issues")
                print("Recommendations:")
                print("- Try: pip install --upgrade py-irt")
                print("- Or: pip install py-irt==1.1.0")
                print("- Check Python version compatibility")
            
            input("\nPress Enter to continue...")
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()