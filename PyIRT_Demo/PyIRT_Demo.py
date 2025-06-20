#!/usr/bin/env python3
"""
Complete Psychometric Testing System using py-irt
=================================================

A comprehensive educational assessment system that:
1. Creates and manages test items
2. Administers tests to students
3. Analyzes responses using Item Response Theory
4. Generates detailed reports

Requirements:
pip install py-irt torch numpy pandas matplotlib seaborn scikit-learn

Author: Psychometric Assessment System
Version: 1.0
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

# Try to import py-irt (if not available, we'll simulate it)
try:
    from pyirt import irt
    PYIRT_AVAILABLE = True
except ImportError:
    print("py-irt not found. Install with: pip install py-irt")
    PYIRT_AVAILABLE = False

class PsychometricItem:
    """Represents a single test item with metadata"""
    
    def __init__(self, item_id: str, question: str, options: List[str], 
                 correct_answer: int, difficulty: float = 0.0, 
                 discrimination: float = 1.0, subject_area: str = "General"):
        self.item_id = item_id
        self.question = question
        self.options = options
        self.correct_answer = correct_answer  # 0-based index
        self.difficulty = difficulty  # IRT difficulty parameter
        self.discrimination = discrimination  # IRT discrimination parameter
        self.subject_area = subject_area
        self.responses = []  # Store all responses to this item
        
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

class Student:
    """Represents a student taking the assessment"""
    
    def __init__(self, student_id: str, name: str, grade_level: str):
        self.student_id = student_id
        self.name = name
        self.grade_level = grade_level
        self.responses = {}  # item_id -> response
        self.ability_estimate = 0.0  # IRT theta estimate
        self.test_start_time = None
        self.test_end_time = None
        
    def record_response(self, item_id: str, response: int):
        """Record student's response to an item"""
        self.responses[item_id] = response
        
    def get_total_score(self, items: Dict[str, PsychometricItem]) -> int:
        """Calculate total raw score"""
        score = 0
        for item_id, response in self.responses.items():
            if item_id in items and items[item_id].is_correct(response):
                score += 1
        return score

class PsychometricTest:
    """Main test management class"""
    
    def __init__(self, test_name: str, test_id: str):
        self.test_name = test_name
        self.test_id = test_id
        self.items = {}  # item_id -> PsychometricItem
        self.students = {}  # student_id -> Student
        self.item_parameters = {}  # IRT parameters
        self.ability_estimates = {}  # Student ability estimates
        
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
        print(f"\n{'='*60}")
        print("Test completed! Thank you for your participation.")
        print(f"Time taken: {student.test_end_time - student.test_start_time}")
        print(f"{'='*60}")
        
        return True
    
    def create_response_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create response matrix for IRT analysis"""
        student_ids = list(self.students.keys())
        item_ids = list(self.items.keys())
        
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
    
    def run_irt_analysis(self):
        """Run IRT analysis on collected responses"""
        print("\n" + "="*60)
        print("RUNNING IRT ANALYSIS")
        print("="*60)
        
        response_matrix, student_ids, item_ids = self.create_response_matrix()
        
        # Check if we have enough data
        if len(student_ids) < 3 or len(item_ids) < 3:
            print("Warning: Insufficient data for reliable IRT analysis.")
            print(f"Students: {len(student_ids)}, Items: {len(item_ids)}")
            return False
            
        # Remove students/items with all missing responses
        student_mask = np.any(response_matrix != -1, axis=1)
        item_mask = np.any(response_matrix != -1, axis=0)
        
        response_matrix = response_matrix[student_mask][:, item_mask]
        student_ids = [sid for i, sid in enumerate(student_ids) if student_mask[i]]
        item_ids = [iid for i, iid in enumerate(item_ids) if item_mask[i]]
        
        print(f"Valid responses: {len(student_ids)} students, {len(item_ids)} items")
        
        if PYIRT_AVAILABLE:
            try:
                # Prepare data for py-irt
                data = []
                for i, student_id in enumerate(student_ids):
                    for j, item_id in enumerate(item_ids):
                        if response_matrix[i, j] != -1:
                            data.append({
                                'user_id': student_id,
                                'item_id': item_id,
                                'outcome': int(response_matrix[i, j])
                            })
                
                # Fit 2PL model
                print("Fitting 2-Parameter Logistic (2PL) IRT model...")
                item_param, user_param = irt(data, theta_bnds=[-4, 4])
                
                # Store parameters
                self.item_parameters = item_param
                self.ability_estimates = user_param
                
                print("IRT Analysis completed successfully!")
                return True
                
            except Exception as e:
                print(f"Error in IRT analysis: {e}")
                print("Falling back to classical test theory analysis...")
                
        # Fallback: Classical Test Theory analysis
        self._classical_analysis(response_matrix, student_ids, item_ids)
        return True
    
    def _classical_analysis(self, response_matrix: np.ndarray, 
                          student_ids: List[str], item_ids: List[str]):
        """Fallback classical test theory analysis"""
        print("Performing Classical Test Theory analysis...")
        
        # Calculate item statistics
        for j, item_id in enumerate(item_ids):
            item_responses = response_matrix[:, j]
            valid_responses = item_responses[item_responses != -1]
            
            if len(valid_responses) > 0:
                difficulty = np.mean(valid_responses)  # p-value
                item = self.items[item_id]
                item.difficulty = difficulty
                item.discrimination = 1.0  # Default discrimination
        
        # Calculate student ability estimates (proportion correct)
        for i, student_id in enumerate(student_ids):
            student_responses = response_matrix[i, :]
            valid_responses = student_responses[student_responses != -1]
            
            if len(valid_responses) > 0:
                ability = np.mean(valid_responses)
                self.students[student_id].ability_estimate = ability
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
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
        report.append(f"Total Students: {len(self.students)}")
        
        # Count completed tests
        completed_tests = sum(1 for s in self.students.values() 
                            if len(s.responses) > 0)
        report.append(f"Completed Tests: {completed_tests}")
        report.append("")
        
        # Student Performance Summary
        report.append("STUDENT PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        if completed_tests > 0:
            scores = []
            for student in self.students.values():
                if len(student.responses) > 0:
                    score = student.get_total_score(self.items)
                    scores.append(score)
                    
            scores = np.array(scores)
            max_score = len(self.items)
            
            report.append(f"Mean Score: {np.mean(scores):.2f} / {max_score}")
            report.append(f"Standard Deviation: {np.std(scores):.2f}")
            report.append(f"Range: {np.min(scores)} - {np.max(scores)}")
            report.append(f"Reliability (Cronbach's α): {self._calculate_alpha():.3f}")
            report.append("")
            
            # Individual student results
            report.append("INDIVIDUAL STUDENT RESULTS")
            report.append("-" * 40)
            report.append(f"{'Student ID':<12} {'Name':<20} {'Score':<8} {'Ability':<10} {'Grade'}")
            report.append("-" * 65)
            
            for student in self.students.values():
                if len(student.responses) > 0:
                    score = student.get_total_score(self.items)
                    ability = getattr(student, 'ability_estimate', 0.0)
                    report.append(f"{student.student_id:<12} {student.name:<20} "
                                f"{score}/{max_score:<6} {ability:<10.3f} {student.grade_level}")
            report.append("")
        
        # Item Analysis
        report.append("ITEM ANALYSIS")
        report.append("-" * 40)
        report.append(f"{'Item ID':<10} {'Subject':<15} {'Difficulty':<12} {'Discrimination':<15} {'Responses'}")
        report.append("-" * 75)
        
        for item_id, item in self.items.items():
            responses_count = len(item.responses)
            difficulty = getattr(item, 'difficulty', 0.0)
            discrimination = getattr(item, 'discrimination', 1.0)
            
            report.append(f"{item_id:<10} {item.subject_area:<15} "
                        f"{difficulty:<12.3f} {discrimination:<15.3f} {responses_count}")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def _calculate_alpha(self) -> float:
        """Calculate Cronbach's alpha reliability coefficient"""
        response_matrix, _, _ = self.create_response_matrix()
        
        # Remove missing data
        valid_matrix = response_matrix[response_matrix != -1].reshape(
            response_matrix.shape[0], -1)
        
        if valid_matrix.size == 0:
            return 0.0
            
        n_items = valid_matrix.shape[1]
        if n_items < 2:
            return 0.0
            
        # Calculate variances
        item_variances = np.var(valid_matrix, axis=0, ddof=1)
        total_variance = np.var(np.sum(valid_matrix, axis=1), ddof=1)
        
        # Cronbach's alpha formula
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
        return max(0.0, alpha)  # Ensure non-negative
    
    def save_results(self, filename: str = None):
        """Save test results to files"""
        if filename is None:
            filename = f"{self.test_id}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save detailed report
        report = self.generate_test_report()
        with open(f"{filename}_report.txt", 'w') as f:
            f.write(report)
        
        # Save raw data as CSV
        response_data = []
        for student_id, student in self.students.items():
            for item_id, response in student.responses.items():
                response_data.append({
                    'student_id': student_id,
                    'student_name': student.name,
                    'grade_level': student.grade_level,
                    'item_id': item_id,
                    'response': response,
                    'correct_answer': self.items[item_id].correct_answer,
                    'is_correct': self.items[item_id].is_correct(response),
                    'subject_area': self.items[item_id].subject_area
                })
        
        if response_data:
            df = pd.DataFrame(response_data)
            df.to_csv(f"{filename}_raw_data.csv", index=False)
        
        print(f"\nResults saved:")
        print(f"- {filename}_report.txt")
        print(f"- {filename}_raw_data.csv")

def create_sample_test() -> PsychometricTest:
    """Create a sample mathematics assessment"""
    test = PsychometricTest("Middle School Mathematics Assessment", "MATH_001")
    
    # Sample math items
    items = [
        PsychometricItem(
            "MATH_001", 
            "What is 15% of 80?",
            ["10", "12", "15", "20"],
            1,  # Correct answer: B (12)
            subject_area="Percentages"
        ),
        PsychometricItem(
            "MATH_002",
            "If x + 7 = 15, what is the value of x?",
            ["6", "7", "8", "9"],
            2,  # Correct answer: C (8)
            subject_area="Algebra"
        ),
        PsychometricItem(
            "MATH_003",
            "What is the area of a rectangle with length 8 cm and width 5 cm?",
            ["13 cm²", "26 cm²", "40 cm²", "45 cm²"],
            2,  # Correct answer: C (40 cm²)
            subject_area="Geometry"
        ),
        PsychometricItem(
            "MATH_004",
            "Which fraction is equivalent to 0.75?",
            ["1/4", "2/3", "3/4", "4/5"],
            2,  # Correct answer: C (3/4)
            subject_area="Fractions"
        ),
        PsychometricItem(
            "MATH_005",
            "What is the mean of the numbers: 4, 6, 8, 10, 12?",
            ["6", "7", "8", "9"],
            2,  # Correct answer: C (8)
            subject_area="Statistics"
        ),
        PsychometricItem(
            "MATH_006",
            "If a triangle has angles of 60° and 70°, what is the third angle?",
            ["40°", "50°", "60°", "70°"],
            1,  # Correct answer: B (50°)
            subject_area="Geometry"
        ),
        PsychometricItem(
            "MATH_007",
            "What is 2³ × 2²?",
            ["16", "24", "32", "64"],
            2,  # Correct answer: C (32)
            subject_area="Exponents"
        ),
        PsychometricItem(
            "MATH_008",
            "Which number is a prime number?",
            ["15", "17", "21", "25"],
            1,  # Correct answer: B (17)
            subject_area="Number Theory"
        )
    ]
    
    # Add items to test
    for item in items:
        test.add_item(item)
    
    return test

def register_sample_students(test: PsychometricTest):
    """Register sample students"""
    students = [
        Student("STU_001", "Sunil Prasath", "Grade 7"),
        Student("STU_002", "Shifa Sonal", "Grade 7"),
        Student("STU_003", "Lucky", "Grade 8"),
        Student("STU_004", "Michael DeSauza", "Grade 8"),
        Student("STU_005", "Franklin", "Grade 7")
    ]
    
    for student in students:
        test.register_student(student)
    
    return students

def main():
    """Main function to run the psychometric testing system"""
    print("="*80)
    print("PSYCHOMETRIC TESTING SYSTEM")
    print("Powered by Item Response Theory (IRT)")
    print("="*80)
    
    # Create sample test
    test = create_sample_test()
    students = register_sample_students(test)
    
    print(f"\nTest created: {test.test_name}")
    print(f"Items loaded: {len(test.items)}")
    print(f"Students registered: {len(test.students)}")
    
    # Main menu
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. View Test Information")
        print("2. Administer Test to Student")
        print("3. Run IRT Analysis")
        print("4. Generate Test Report")
        print("5. Save Results")
        print("6. Simulate Student Responses (Demo)")
        print("0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == "0":
            print("Thank you for using the Psychometric Testing System!")
            break
            
        elif choice == "1":
            print(f"\nTest: {test.test_name}")
            print(f"Items: {len(test.items)}")
            print(f"Students: {len(test.students)}")
            print("\nRegistered Students:")
            for student in test.students.values():
                responses_count = len(student.responses)
                status = "Completed" if responses_count > 0 else "Not Started"
                print(f"  {student.student_id}: {student.name} ({student.grade_level}) - {status}")
                
        elif choice == "2":
            print("\nAvailable Students:")
            for student in test.students.values():
                status = "Completed" if len(student.responses) > 0 else "Available"
                print(f"  {student.student_id}: {student.name} - {status}")
            
            student_id = input("\nEnter Student ID: ").strip()
            if student_id in test.students:
                if len(test.students[student_id].responses) > 0:
                    retry = input("Student has already taken the test. Retake? (y/N): ")
                    if retry.lower() != 'y':
                        continue
                    test.students[student_id].responses.clear()
                
                test.administer_test(student_id)
            else:
                print("Invalid Student ID!")
                
        elif choice == "3":
            completed_tests = sum(1 for s in test.students.values() if len(s.responses) > 0)
            if completed_tests < 2:
                print(f"Need at least 2 completed tests for analysis. Current: {completed_tests}")
            else:
                test.run_irt_analysis()
                
        elif choice == "4":
            print("\n" + test.generate_test_report())
            
        elif choice == "5":
            filename = input("Enter filename prefix (or press Enter for default): ").strip()
            if not filename:
                filename = None
            test.save_results(filename)
            
        elif choice == "6":
            print("Simulating student responses for demonstration...")
            
            # Simulate responses for all students
            np.random.seed(42)  # For reproducible results
            
            for student in test.students.values():
                if len(student.responses) == 0:  # Only simulate if not already completed
                    # Simulate ability-based responses
                    ability = np.random.normal(0, 1)  # Random ability
                    student.ability_estimate = ability
                    
                    for item_id, item in test.items.items():
                        # Simple IRT probability model for simulation
                        prob_correct = 1 / (1 + np.exp(-(ability - item.difficulty)))
                        is_correct = np.random.random() < prob_correct
                        
                        if is_correct:
                            response = item.correct_answer
                        else:
                            # Random incorrect response
                            options = list(range(len(item.options)))
                            options.remove(item.correct_answer)
                            response = np.random.choice(options)
                        
                        student.record_response(item_id, response)
                        item.add_response(student.student_id, response, is_correct)
            
            print("Simulation completed! All students now have responses.")
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()