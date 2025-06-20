#!/usr/bin/env python3
"""
LLM Student Simulator for Psychometric Testing
==============================================

This script uses Ollama with gemma3:12b to simulate realistic student responses
for a comprehensive psychometric assessment across multiple cognitive domains.

Features:
- Math, Verbal, and Spatial Reasoning questions (10 each)
- Realistic LLM-based student simulation with varying ability levels
- Integration with PyIRT_3LP.py for advanced psychometric analysis
- Comprehensive test administration and analysis

Requirements:
pip install ollama requests numpy pandas

Usage:
python llm_student_simulator.py

Author: LLM-Enhanced Psychometric Testing System
Version: 1.1 (Fixed Ollama Integration)
"""

import sys
import os
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import ollama client
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("✓ Ollama client available")
except ImportError:
    print("✗ Ollama not found. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False
    sys.exit(1)

# Import the PyIRT_3LP module
try:
    from PyIRT_3LP import (
        PsychometricItem, 
        Student, 
        EnhancedPsychometricTest,
        simulate_realistic_responses
    )
    print("✓ PyIRT_3LP module imported successfully")
except ImportError:
    print("✗ PyIRT_3LP.py not found in current directory")
    print("Please ensure PyIRT_3LP.py is in the same folder as this script")
    sys.exit(1)

class LLMStudent:
    """Represents a virtual student powered by LLM"""
    
    def __init__(self, student_id: str, name: str, grade_level: str, 
                 cognitive_profile: str = "average", model_name: str = "gemma3:12b"):
        self.student_id = student_id
        self.name = name
        self.grade_level = grade_level
        self.cognitive_profile = cognitive_profile  # "low", "average", "high", "gifted"
        self.model_name = model_name
        
        # Response patterns and tendencies
        self.response_time_factor = self._get_response_time_factor()
        self.consistency_level = self._get_consistency_level()
        self.question_count = 0
        
    def _get_response_time_factor(self) -> float:
        """Get response time multiplier based on cognitive profile"""
        factors = {
            "low": 1.5,      # Slower responses
            "average": 1.0,   # Normal speed
            "high": 0.8,     # Faster responses  
            "gifted": 0.6    # Very fast responses
        }
        return factors.get(self.cognitive_profile, 1.0)
    
    def _get_consistency_level(self) -> float:
        """Get consistency level (0.0-1.0) based on cognitive profile"""
        levels = {
            "low": 0.6,      # Less consistent
            "average": 0.8,   # Moderately consistent
            "high": 0.9,     # Very consistent
            "gifted": 0.95   # Extremely consistent
        }
        return levels.get(self.cognitive_profile, 0.8)
    
    def create_prompt(self, item: PsychometricItem) -> str:
        """Create a contextual prompt for the LLM based on student profile"""
        
        # Base cognitive profile descriptions
        profile_descriptions = {
            "low": "You are a student who struggles with academic work. You often make mistakes, get confused easily, and sometimes guess when unsure. You might misread questions or make computational errors.",
            "average": "You are a typical student with average academic ability. You understand most concepts but sometimes make mistakes. You work through problems methodically but may struggle with complex questions.",
            "high": "You are a strong student with above-average academic ability. You understand concepts well, work systematically, and rarely make careless mistakes. You can handle challenging problems.",
            "gifted": "You are an exceptional student with superior academic ability. You quickly grasp complex concepts, see patterns easily, and solve problems efficiently. You rarely make mistakes."
        }
        
        # Subject-specific adjustments
        subject_adjustments = {
            "Mathematics": {
                "low": "Math is particularly challenging for you. You often struggle with calculations and may confuse formulas.",
                "average": "You have a decent grasp of math concepts but sometimes make computational errors.",
                "high": "You're quite good at math and can work through most problems systematically.",
                "gifted": "Mathematics comes naturally to you. You can quickly identify the right approach and execute it accurately."
            },
            "Verbal Reasoning": {
                "low": "Reading comprehension and vocabulary are difficult for you. You sometimes misunderstand passages or word meanings.",
                "average": "You have reasonable reading skills but may struggle with complex texts or subtle meanings.",
                "high": "You're good with language, understand context well, and have a solid vocabulary.",
                "gifted": "You excel with language, quickly grasp subtle meanings, and have an extensive vocabulary."
            },
            "Spatial Reasoning": {
                "low": "Visualizing shapes and spatial relationships is very difficult for you. You often get confused by rotations and transformations.",
                "average": "You can handle basic spatial problems but struggle with complex 3D visualizations.",
                "high": "You're quite good at visualizing shapes and understanding spatial relationships.",
                "gifted": "Spatial visualization is one of your strengths. You can easily manipulate mental images and see patterns."
            }
        }
        
        # Create the prompt
        base_description = profile_descriptions[self.cognitive_profile]
        subject_description = subject_adjustments.get(item.subject_area, {}).get(self.cognitive_profile, "")
        
        prompt = f"""You are taking a standardized test. Here's your profile:

{base_description}

Subject-specific note: {subject_description}

Grade Level: {self.grade_level}
Current Question #{self.question_count + 1}

Question: {item.question}

Options:
"""
        
        for i, option in enumerate(item.options):
            prompt += f"{chr(65+i)}. {option}\n"
        
        prompt += f"""
Instructions:
- Read the question carefully
- Consider each option
- Choose the SINGLE BEST answer based on your ability level
- Respond with ONLY the letter (A, B, C, or D)
- Do not explain your reasoning
- Be consistent with your academic profile ({self.cognitive_profile})

Your answer:"""
        
        return prompt
    
    def get_llm_response(self, prompt: str) -> str:
        """Get response from Ollama LLM with proper error handling"""
        try:
            # Add some response time simulation
            base_time = 2.0 * self.response_time_factor
            time.sleep(random.uniform(0.5, base_time))
            
            # Try the generate call
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 5,
                    'top_p': 0.9
                }
            )
            
            # Handle the Ollama GenerateResponse object
            response_text = None
            
            # Check if it's an Ollama GenerateResponse object
            if hasattr(response, 'response'):
                response_text = response.response
            elif isinstance(response, dict):
                if 'response' in response:
                    response_text = response['response']
                elif 'content' in response:
                    response_text = response['content']
                elif 'text' in response:
                    response_text = response['text']
                else:
                    raise ValueError(f"Unexpected dict response structure. Keys: {response.keys()}")
            elif isinstance(response, str):
                response_text = response
            else:
                raise ValueError(f"Unexpected response type: {type(response)}. Response: {response}")
            
            if response_text is None:
                raise ValueError("Could not extract response text from Ollama response")
                
            return response_text.strip()
                
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR: LLM response failed for {self.name}")
            print(f"Error: {e}")
            print(f"Model: {self.model_name}")
            print(f"Response type: {type(response) if 'response' in locals() else 'No response'}")
            if 'response' in locals():
                print(f"Response attributes: {dir(response)}")
            
            # Try alternative model names if the primary fails
            alternative_models = ["llama3.2", "llama3.2:latest"]
            for alt_model in alternative_models:
                try:
                    print(f"Attempting fallback with model: {alt_model}")
                    fallback_response = ollama.generate(
                        model=alt_model,
                        prompt=prompt,
                        options={'temperature': 0.3, 'num_predict': 5}
                    )
                    
                    # Try to extract response from fallback
                    if hasattr(fallback_response, 'response'):
                        return fallback_response.response.strip()
                    elif isinstance(fallback_response, dict) and 'response' in fallback_response:
                        return fallback_response['response'].strip()
                    
                except Exception as fallback_error:
                    print(f"Fallback model {alt_model} also failed: {fallback_error}")
                    continue
            
            # If all models fail, raise an error instead of random choice
            raise RuntimeError(f"All LLM models failed for student {self.name}. Cannot continue simulation without valid responses.")
    
    def answer_question(self, item: PsychometricItem) -> int:
        """Answer a question using LLM with cognitive profile simulation"""
        self.question_count += 1
        
        try:
            # Create contextual prompt
            prompt = self.create_prompt(item)
            
            # Get LLM response
            llm_response = self.get_llm_response(prompt)
            
            # Extract answer
            answer_letter = self._extract_answer(llm_response)
            answer_index = ord(answer_letter) - ord('A')
            
            # Validate answer index
            if not (0 <= answer_index < len(item.options)):
                raise ValueError(f"Invalid answer index {answer_index} for question with {len(item.options)} options")
            
            # Apply consistency check (sometimes override LLM for profile consistency)
            if random.random() > self.consistency_level:
                # Less consistent students might change their answer or make mistakes
                if self.cognitive_profile == "low":
                    # Low ability students might randomly guess more often
                    if random.random() < 0.3:
                        answer_index = random.randint(0, len(item.options) - 1)
                        answer_letter = chr(ord('A') + answer_index)
                        print(f"  {self.name}: Consistency override -> {answer_letter}")
            
            print(f"  {self.name} ({self.cognitive_profile}): Q{self.question_count} -> {answer_letter} (LLM: '{llm_response.strip()}')")
            
            return answer_index
            
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR: Failed to get answer from {self.name} for question {self.question_count}")
            print(f"Error: {e}")
            print(f"Question: {item.question}")
            print("SIMULATION STOPPED - Cannot continue without valid LLM responses")
            raise RuntimeError(f"Student {self.name} failed to answer question {self.question_count}: {e}")
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer choice from LLM response"""
        if not response or not isinstance(response, str):
            raise ValueError(f"Invalid response for answer extraction: {response} (type: {type(response)})")
        
        # Look for A, B, C, or D in the response
        import re
        pattern = r'\b([ABCD])\b'
        matches = re.findall(pattern, response.upper())
        
        if matches:
            return matches[0]
        
        # If no clear answer found, raise error instead of random choice
        raise ValueError(f"Could not extract valid answer (A, B, C, or D) from LLM response: '{response}'")

def create_comprehensive_test() -> EnhancedPsychometricTest:
    """Create a comprehensive test with Math, Verbal, and Spatial Reasoning questions"""
    
    test = EnhancedPsychometricTest("Comprehensive Cognitive Assessment", "CCA_LLM_001")
    
    # Mathematics Questions (10)
    math_questions = [
        PsychometricItem(
            "MATH_001", 
            "What is 25% of 120?",
            ["25", "30", "35", "40"],
            1,  # B: 30
            difficulty=-0.3, discrimination=1.2, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_002",
            "If 3x + 7 = 22, what is x?",
            ["3", "5", "7", "9"],
            1,  # B: 5
            difficulty=0.1, discrimination=1.4, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_003",
            "What is the area of a triangle with base 8 cm and height 6 cm?",
            ["24 cm²", "28 cm²", "32 cm²", "48 cm²"],
            0,  # A: 24 cm²
            difficulty=0.2, discrimination=1.3, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_004",
            "Which fraction is equivalent to 0.6?",
            ["2/3", "3/5", "5/8", "7/12"],
            1,  # B: 3/5
            difficulty=-0.1, discrimination=1.1, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_005",
            "What is the median of: 3, 7, 5, 9, 5, 4, 8?",
            ["4", "5", "6", "7"],
            1,  # B: 5
            difficulty=0.4, discrimination=1.2, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_006",
            "If a rectangle has length 12 and width 8, what is its perimeter?",
            ["32", "40", "48", "96"],
            1,  # B: 40
            difficulty=-0.2, discrimination=1.0, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_007",
            "What is 2⁴ × 2³?",
            ["2⁷", "2¹²", "4⁷", "4¹²"],
            0,  # A: 2⁷
            difficulty=0.6, discrimination=1.5, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_008",
            "Which number is a perfect square?",
            ["18", "24", "36", "42"],
            2,  # C: 36
            difficulty=0.0, discrimination=1.1, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_009",
            "If y = 2x + 3 and x = 4, what is y?",
            ["9", "11", "13", "15"],
            1,  # B: 11
            difficulty=0.3, discrimination=1.3, guessing=0.25,
            subject_area="Mathematics"
        ),
        PsychometricItem(
            "MATH_010",
            "What is the volume of a cube with side length 3?",
            ["9", "18", "27", "36"],
            2,  # C: 27
            difficulty=0.5, discrimination=1.4, guessing=0.25,
            subject_area="Mathematics"
        )
    ]
    
    # Verbal Reasoning Questions (10)
    verbal_questions = [
        PsychometricItem(
            "VERB_001",
            "Choose the word that best completes the analogy: Cat is to Meow as Dog is to ___",
            ["Run", "Bark", "Walk", "Jump"],
            1,  # B: Bark
            difficulty=-0.4, discrimination=1.1, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_002",
            "Which word is most similar in meaning to 'benevolent'?",
            ["Kind", "Angry", "Confused", "Tired"],
            0,  # A: Kind
            difficulty=0.2, discrimination=1.3, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_003",
            "Choose the word that does NOT belong: Apple, Orange, Carrot, Banana",
            ["Apple", "Orange", "Carrot", "Banana"],
            2,  # C: Carrot
            difficulty=0.0, discrimination=1.2, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_004",
            "What is the opposite of 'sparse'?",
            ["Dense", "Empty", "Clear", "Bright"],
            0,  # A: Dense
            difficulty=0.4, discrimination=1.4, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_005",
            "Complete the sequence: Monday, Wednesday, Friday, ___",
            ["Saturday", "Sunday", "Tuesday", "Thursday"],
            1,  # B: Sunday
            difficulty=0.1, discrimination=1.0, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_006",
            "Which word best describes someone who is 'meticulous'?",
            ["Careful", "Careless", "Fast", "Loud"],
            0,  # A: Careful
            difficulty=0.3, discrimination=1.2, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_007",
            "Book is to Read as Music is to ___",
            ["Write", "Listen", "See", "Touch"],
            1,  # B: Listen
            difficulty=-0.1, discrimination=1.1, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_008",
            "Which word means 'to make less severe'?",
            ["Aggravate", "Mitigate", "Complicate", "Accelerate"],
            1,  # B: Mitigate
            difficulty=0.7, discrimination=1.5, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_009",
            "Choose the correct sentence:",
            ["Him and I went to the store", "He and I went to the store", "Me and him went to the store", "I and him went to the store"],
            1,  # B: He and I went to the store
            difficulty=0.2, discrimination=1.3, guessing=0.25,
            subject_area="Verbal Reasoning"
        ),
        PsychometricItem(
            "VERB_010",
            "What does 'ubiquitous' mean?",
            ["Rare", "Everywhere", "Ancient", "Expensive"],
            1,  # B: Everywhere
            difficulty=0.6, discrimination=1.4, guessing=0.25,
            subject_area="Verbal Reasoning"
        )
    ]
    
    # Spatial Reasoning Questions (10)
    spatial_questions = [
        PsychometricItem(
            "SPAT_001",
            "If you rotate a square 90 degrees clockwise, what shape do you get?",
            ["Rectangle", "Square", "Triangle", "Circle"],
            1,  # B: Square
            difficulty=-0.2, discrimination=1.0, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_002",
            "Which shape can be formed by folding this net: six connected squares in a cross pattern?",
            ["Pyramid", "Cube", "Cylinder", "Cone"],
            1,  # B: Cube
            difficulty=0.1, discrimination=1.2, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_003",
            "If you look at a circle from the side, what shape do you see?",
            ["Circle", "Line", "Square", "Triangle"],
            1,  # B: Line
            difficulty=0.0, discrimination=1.1, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_004",
            "How many faces does a cube have?",
            ["4", "6", "8", "12"],
            1,  # B: 6
            difficulty=-0.1, discrimination=1.0, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_005",
            "If you stack 3 cubes on top of each other, how many total faces are visible from the outside?",
            ["14", "16", "18", "20"],
            0,  # A: 14
            difficulty=0.5, discrimination=1.4, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_006",
            "Which 2D shape has the most sides?",
            ["Triangle", "Square", "Pentagon", "Hexagon"],
            3,  # D: Hexagon
            difficulty=-0.3, discrimination=0.9, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_007",
            "If you cut a sphere in half, what shape is each half?",
            ["Circle", "Hemisphere", "Oval", "Cylinder"],
            1,  # B: Hemisphere
            difficulty=0.2, discrimination=1.1, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_008",
            "Which shape cannot be made by cutting a cube?",
            ["Triangle", "Rectangle", "Pentagon", "Square"],
            2,  # C: Pentagon
            difficulty=0.4, discrimination=1.3, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_009",
            "If you unfold a pyramid with a square base, how many triangular faces will you see?",
            ["2", "3", "4", "5"],
            2,  # C: 4
            difficulty=0.3, discrimination=1.2, guessing=0.25,
            subject_area="Spatial Reasoning"
        ),
        PsychometricItem(
            "SPAT_010",
            "Which object has rotational symmetry?",
            ["Letter F", "Letter L", "Circle", "Letter P"],
            2,  # C: Circle
            difficulty=0.1, discrimination=1.1, guessing=0.25,
            subject_area="Spatial Reasoning"
        )
    ]
    
    # Add all questions to test
    all_questions = math_questions + verbal_questions + spatial_questions
    for item in all_questions:
        test.add_item(item)
    
    return test

def create_llm_students(num_students: int) -> List[LLMStudent]:
    """Create diverse LLM students with different cognitive profiles"""
    
    students = []
    
    # Define cognitive profile distribution
    profile_distribution = {
        "low": 0.15,      # 15% low ability
        "average": 0.60,  # 60% average ability  
        "high": 0.20,     # 20% high ability
        "gifted": 0.05    # 5% gifted
    }
    
    # Generate names
    first_names = [
        "Alex", "Jamie", "Taylor", "Jordan", "Casey", "Riley", "Avery", "Quinn",
        "Morgan", "Sage", "River", "Sky", "Phoenix", "Rowan", "Emery", "Drew",
        "Blake", "Cameron", "Hayden", "Peyton", "Reese", "Parker", "Charlie", "Devon"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"
    ]
    
    grade_levels = ["Grade 7", "Grade 8", "Grade 9"]
    
    for i in range(num_students):
        # Determine cognitive profile based on distribution
        rand = random.random()
        cumulative = 0
        profile = "average"
        
        for prof, prob in profile_distribution.items():
            cumulative += prob
            if rand <= cumulative:
                profile = prof
                break
        
        # Generate student details
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        name = f"{first_name} {last_name}"
        grade = random.choice(grade_levels)
        student_id = f"LLM_{i+1:03d}"
        
        student = LLMStudent(
            student_id=student_id,
            name=name,
            grade_level=grade,
            cognitive_profile=profile
        )
        
        students.append(student)
    
    return students

def check_ollama_connection() -> bool:
    """Check if Ollama is running and gemma3:12b is available"""
    try:
        # First, test basic connectivity
        try:
            models_response = ollama.list()
            print(f"✓ Ollama is running")
        except Exception as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            return False
        
        # Handle the Ollama ListResponse object
        models_list = []
        
        try:
            # The ListResponse object should have a 'models' attribute or be iterable
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif hasattr(models_response, '__iter__'):
                models_list = list(models_response)
            else:
                # Try to convert to dict and access models
                models_dict = dict(models_response) if hasattr(models_response, '__dict__') else {}
                models_list = models_dict.get('models', [])
        except Exception as e:
            print(f"Debug - Error accessing models: {e}")
            # Try alternative approaches
            try:
                # Convert to string and parse, or access via attributes
                models_list = getattr(models_response, 'models', [])
            except:
                models_list = []
        
        print(f"✓ Found {len(models_list)} models")
        
        # Extract model names safely
        model_names = []
        for model in models_list:
            try:
                if hasattr(model, 'name'):
                    model_names.append(model.name)
                elif isinstance(model, dict):
                    # Try different possible keys for model name
                    name = model.get('name') or model.get('model') or model.get('id')
                    if name:
                        model_names.append(name)
                elif isinstance(model, str):
                    model_names.append(model)
                else:
                    # Try to get name attribute or convert to string
                    name = getattr(model, 'name', str(model))
                    model_names.append(name)
            except Exception as e:
                print(f"Debug - Error processing model: {e}")
                continue
        
        print(f"Available models: {model_names}")
        
        # Check for the specific model we need
        target_models = ["gemma3:12b", "llama3.2", "llama3.2:latest"]
        found_model = None
        
        for target in target_models:
            # Check for exact match or partial match
            for available in model_names:
                if target in available or available in target:
                    found_model = available
                    break
            if found_model:
                break
        
        if found_model:
            print(f"✓ Found compatible model: {found_model}")
            return True
        else:
            print("✗ gemma3:12b model not found")
            print("Please run one of these commands:")
            print("  ollama pull gemma3:12b")
            print("  ollama pull llama3.2")
            return False
            
    except Exception as e:
        print(f"✗ Unexpected error checking Ollama: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_basic() -> bool:
    """Basic test to ensure Ollama is working"""
    try:
        print("Testing basic Ollama connectivity...")
        
        # Try a simple generate call
        test_response = ollama.generate(
            model="gemma3:12b",
            prompt="Answer with just the letter A:",
            options={'num_predict': 1}
        )
        
        print("✓ Basic Ollama test successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic Ollama test failed: {e}")
        
        # Try with alternative model name
        try:
            test_response = ollama.generate(
                model="llama3.2",
                prompt="Answer with just the letter A:",
                options={'num_predict': 1}
            )
            print("✓ Basic Ollama test successful with alternative model")
            return True
        except Exception as e2:
            print(f"✗ Alternative model test also failed: {e2}")
            return False

def run_llm_simulation(test: EnhancedPsychometricTest, llm_students: List[LLMStudent]):
    """Run the LLM-based test simulation"""
    
    print(f"\n{'='*80}")
    print("STARTING LLM-BASED PSYCHOMETRIC TESTING")
    print(f"{'='*80}")
    print(f"Test: {test.test_name}")
    print(f"Total Questions: {len(test.items)}")
    print(f"Virtual Students: {len(llm_students)}")
    print(f"Model: gemma3:12b")
    
    # Convert LLM students to PyIRT Student objects and register them
    for llm_student in llm_students:
        pyirt_student = Student(
            student_id=llm_student.student_id,
            name=llm_student.name,
            grade_level=llm_student.grade_level
        )
        test.register_student(pyirt_student)
    
    print(f"\n✓ {len(llm_students)} students registered in test system")
    
    # Simulate test administration
    print(f"\nAdministering test to LLM students...")
    print("This may take several minutes depending on the number of students...")
    
    start_time = datetime.now()

    
    for i, llm_student in enumerate(llm_students):
        print(f"\nStudent {i+1}/{len(llm_students)}: {llm_student.name} ({llm_student.cognitive_profile})")
        
        # Get corresponding PyIRT student
        pyirt_student = test.students[llm_student.student_id]
        pyirt_student.test_start_time = datetime.now()
        
        # Answer all questions
        for item_id, item in test.items.items():
            response_idx = llm_student.answer_question(item)
            
            # Record response in PyIRT system
            pyirt_student.record_response(item_id, response_idx)
            item.add_response(llm_student.student_id, response_idx, 
                            item.is_correct(response_idx))
        
        # Calculate performance metrics
        pyirt_student.test_end_time = datetime.now()
        pyirt_student.calculate_performance_metrics(test.items)
        
        print(f"  Completed: {pyirt_student.raw_score}/{len(test.items)} correct ({pyirt_student.percent_correct:.1f}%)")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n✓ Test administration completed in {total_time}")
    print(f"✓ All {len(llm_students)} students have completed the assessment")

def main():
    """Main function to run the LLM-based psychometric testing"""
    
    print("LLM-Based Psychometric Testing System")
    print("=" * 50)
    
    # Check Ollama connection
    if not check_ollama_connection():
        return
    
    # Get number of students
    while True:
        try:
            num_students = input("\nEnter number of virtual students (10-50, default 20): ").strip()
            if not num_students:
                num_students = 20
            else:
                num_students = int(num_students)
            
            if 1 <= num_students <= 50:
                break
            else:
                print("Please enter a number between 1 and 50")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nGenerating {num_students} virtual students...")
    
    # Create test and students
    test = create_comprehensive_test()
    llm_students = create_llm_students(num_students)
    
    # Show student distribution
    profile_counts = {}
    for student in llm_students:
        profile_counts[student.cognitive_profile] = profile_counts.get(student.cognitive_profile, 0) + 1
    
    print(f"\nStudent Cognitive Profile Distribution:")
    for profile, count in profile_counts.items():
        percentage = (count / num_students) * 100
        print(f"  {profile.capitalize()}: {count} students ({percentage:.1f}%)")
    
    # Confirm before starting
    print(f"\nAbout to administer 30-question test to {num_students} LLM students")
    print("This will make approximately {} LLM calls and may take 10-30 minutes".format(num_students * 30))
    
    confirm = input("\nProceed with LLM simulation? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Simulation cancelled.")
        return
    
    # Run the simulation
    run_llm_simulation(test, llm_students)
    
    # Run psychometric analysis
    print(f"\n{'='*60}")
    print("RUNNING PSYCHOMETRIC ANALYSIS")
    print(f"{'='*60}")
    
    success = test.run_irt_analysis()
    
    if success:
        print(f"\n✓ IRT Analysis completed successfully")
        print(f"  Model: {test.irt_analysis_message}")
        print(f"  Items analyzed: {len(test.item_parameters)}")
        print(f"  Students analyzed: {len(test.ability_estimates)}")
        print(f"  Reliability (α): {test.reliability_alpha:.3f}")
    else:
        print(f"\n⚠ IRT Analysis failed: {test.irt_analysis_message}")
        print("Classical Test Theory analysis was used instead")
    
    # Generate and display report
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    report = test.generate_test_report()
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"LLM_Psychometric_Test_{timestamp}"
    
    print(f"\nSaving results...")
    test.save_results(filename)
    
    print(f"\n{'='*80}")
    print("LLM PSYCHOMETRIC TESTING COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results saved with prefix: {filename}")
    print(f"Students tested: {num_students}")
    print(f"Questions administered: {len(test.items)}")
    print(f"Total responses collected: {sum(len(s.responses) for s in test.students.values())}")
    
    # Show performance by cognitive profile
    print(f"\nPerformance by Cognitive Profile:")
    profile_performance = {}
    
    for llm_student in llm_students:
        profile = llm_student.cognitive_profile
        pyirt_student = test.students[llm_student.student_id]
        
        if profile not in profile_performance:
            profile_performance[profile] = []
        profile_performance[profile].append(pyirt_student.percent_correct)
    
    for profile, scores in profile_performance.items():
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        count = len(scores)
        print(f"  {profile.capitalize()}: {avg_score:.1f}% ± {std_score:.1f}% (n={count})")
    
    # Show subject area performance
    print(f"\nPerformance by Subject Area:")
    subject_performance = {
        "Mathematics": [],
        "Verbal Reasoning": [], 
        "Spatial Reasoning": []
    }
    
    for student in test.students.values():
        if len(student.responses) == 0:
            continue
            
        subject_scores = {"Mathematics": 0, "Verbal Reasoning": 0, "Spatial Reasoning": 0}
        subject_totals = {"Mathematics": 0, "Verbal Reasoning": 0, "Spatial Reasoning": 0}
        
        for item_id, response in student.responses.items():
            if item_id in test.items:
                item = test.items[item_id]
                subject_totals[item.subject_area] += 1
                if item.is_correct(response):
                    subject_scores[item.subject_area] += 1
        
        for subject in subject_scores:
            if subject_totals[subject] > 0:
                percentage = (subject_scores[subject] / subject_totals[subject]) * 100
                subject_performance[subject].append(percentage)
    
    for subject, scores in subject_performance.items():
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {subject}: {avg_score:.1f}% ± {std_score:.1f}%")

def analyze_llm_response_patterns(test: EnhancedPsychometricTest, llm_students: List[LLMStudent]):
    """Analyze LLM response patterns for insights"""
    
    print(f"\n{'='*60}")
    print("LLM RESPONSE PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze response distribution by difficulty
    difficulty_analysis = {"Easy": [], "Medium": [], "Hard": []}
    
    for item_id, item in test.items.items():
        if len(item.responses) == 0:
            continue
            
        correct_responses = sum(1 for r in item.responses if r['is_correct'])
        total_responses = len(item.responses)
        p_value = correct_responses / total_responses if total_responses > 0 else 0
        
        if p_value >= 0.7:
            difficulty_analysis["Easy"].append(p_value)
        elif p_value >= 0.4:
            difficulty_analysis["Medium"].append(p_value)
        else:
            difficulty_analysis["Hard"].append(p_value)
    
    print("Item Difficulty Distribution:")
    for difficulty, p_values in difficulty_analysis.items():
        if p_values:
            avg_p = np.mean(p_values)
            count = len(p_values)
            print(f"  {difficulty}: {count} items, avg p-value = {avg_p:.3f}")
    
    # Analyze cognitive profile accuracy
    print(f"\nCognitive Profile Validation:")
    expected_performance = {
        "low": (0.3, 0.5),      # 30-50% expected
        "average": (0.5, 0.7),   # 50-70% expected
        "high": (0.7, 0.85),    # 70-85% expected
        "gifted": (0.85, 0.95)  # 85-95% expected
    }
    
    for llm_student in llm_students:
        pyirt_student = test.students[llm_student.student_id]
        actual_performance = pyirt_student.percent_correct / 100
        expected_min, expected_max = expected_performance[llm_student.cognitive_profile]
        
        if expected_min <= actual_performance <= expected_max:
            status = "✓ As Expected"
        elif actual_performance < expected_min:
            status = "↓ Below Expected"
        else:
            status = "↑ Above Expected"
        
        print(f"  {llm_student.name} ({llm_student.cognitive_profile}): {actual_performance:.1%} {status}")

def create_detailed_summary():
    """Create a detailed summary of the LLM testing approach"""
    
    summary = """
    
═══════════════════════════════════════════════════════════════════════════════
                    LLM-BASED PSYCHOMETRIC TESTING SUMMARY
═══════════════════════════════════════════════════════════════════════════════

METHODOLOGY:
• Used gemma3:12b model via Ollama to simulate realistic student responses
• Created diverse cognitive profiles (low, average, high, gifted) with different
  response patterns and consistency levels
• Administered 30 questions across 3 cognitive domains:
  - Mathematics (10 questions): Arithmetic, algebra, geometry
  - Verbal Reasoning (10 questions): Analogies, vocabulary, logic
  - Spatial Reasoning (10 questions): 3D visualization, patterns

COGNITIVE SIMULATION:
• Each LLM student has contextual prompts based on their ability level
• Response times and consistency vary by cognitive profile
• Profile-specific descriptions influence answer quality and patterns

PSYCHOMETRIC ANALYSIS:
• Item Response Theory (IRT) analysis using multiple libraries (girth, pyirt)
• 3PL model attempts with fallback to 2PL and 1PL models
• Classical Test Theory backup for comprehensive analysis
• Reliability analysis using Cronbach's alpha

VALIDATION APPROACH:
• Compare LLM performance to expected cognitive profile ranges
• Analyze item difficulty patterns and discrimination
• Subject area performance analysis
• Response pattern validation against psychometric theory

ADVANTAGES:
• Realistic response simulation at scale
• Consistent testing conditions
• Diverse ability representation
• Reproducible results for validation
• Cost-effective large-scale testing

This approach demonstrates the potential for LLM-based psychometric validation
while providing insights into both assessment quality and AI response patterns.
    """
    
    return summary

if __name__ == "__main__":
    try:
        main()
        
        # Print detailed summary
        print(create_detailed_summary())
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()