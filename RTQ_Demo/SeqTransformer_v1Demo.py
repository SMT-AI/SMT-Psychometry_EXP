import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from typing import List, Tuple, Dict
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Question:
    def __init__(self, id: int, text: str, options: List[str], correct_answer: int, difficulty: str):
        self.id = id
        self.text = text
        self.options = options
        self.correct_answer = correct_answer
        self.difficulty = difficulty
        self.difficulty_level = {'Easy': 0, 'Medium': 1, 'Hard': 2}[difficulty]

class StudentInteraction:
    def __init__(self, question_id: int, difficulty_level: int, is_correct: bool, response_time: float):
        self.question_id = question_id
        self.difficulty_level = difficulty_level
        self.is_correct = is_correct
        self.response_time = response_time

class TransformerQuestionSelector(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=8, num_layers=4, max_seq_len=50):
        super(TransformerQuestionSelector, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embeddings for different features
        self.question_embedding = nn.Embedding(vocab_size, d_model // 4)
        self.difficulty_embedding = nn.Embedding(3, d_model // 4)  # Easy, Medium, Hard
        self.correctness_embedding = nn.Embedding(2, d_model // 4)  # Correct, Incorrect
        self.time_projection = nn.Linear(1, d_model // 4)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for next question selection
        self.difficulty_predictor = nn.Linear(d_model, 3)  # Predict next difficulty
        self.confidence_predictor = nn.Linear(d_model, 1)  # Predict confidence in selection
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, question_ids, difficulty_levels, correctness, response_times, mask=None):
        batch_size, seq_len = question_ids.shape
        
        # Create embeddings
        q_emb = self.question_embedding(question_ids)
        d_emb = self.difficulty_embedding(difficulty_levels)
        c_emb = self.correctness_embedding(correctness)
        t_emb = self.time_projection(response_times.unsqueeze(-1))
        
        # Concatenate embeddings
        x = torch.cat([q_emb, d_emb, c_emb, t_emb], dim=-1)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)
        
        # Apply transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Get the last sequence element for prediction
        last_hidden = x[:, -1, :]
        
        # Predict next difficulty and confidence
        difficulty_logits = self.difficulty_predictor(last_hidden)
        confidence = torch.sigmoid(self.confidence_predictor(last_hidden))
        
        return difficulty_logits, confidence

class AdaptiveAssessmentSystem:
    def __init__(self):
        self.questions = self._initialize_questions()
        self.model = TransformerQuestionSelector()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.interaction_history = []
        self.current_student_history = []
        
        # Simulate some training data
        self._simulate_training_data()
        self._train_model()
        
    def _initialize_questions(self) -> Dict[str, List[Question]]:
        questions = {
            'Easy': [
                Question(1, "What is 2 + 2?", ["3", "4", "5", "6"], 1, "Easy"),
                Question(2, "What color do you get when you mix red and blue?", ["Green", "Purple", "Orange", "Yellow"], 1, "Easy"),
                Question(3, "How many days are in a week?", ["5", "6", "7", "8"], 2, "Easy"),
                Question(4, "What is the capital of France?", ["London", "Berlin", "Paris", "Madrid"], 2, "Easy"),
            ],
            'Medium': [
                Question(5, "What is 15 × 7?", ["95", "105", "115", "125"], 1, "Medium"),
                Question(6, "Which planet is closest to the Sun?", ["Venus", "Mercury", "Earth", "Mars"], 1, "Medium"),
                Question(7, "What is the square root of 64?", ["6", "7", "8", "9"], 2, "Medium"),
            ],
            'Hard': [
                Question(8, "What is the derivative of x² + 3x?", ["2x + 3", "x + 3", "2x", "3x"], 0, "Hard"),
                Question(9, "Which of these is a prime number?", ["21", "27", "29", "33"], 2, "Hard"),
                Question(10, "In which year did World War II end?", ["1944", "1945", "1946", "1947"], 1, "Hard"),
            ]
        }
        return questions
    
    def _simulate_training_data(self, num_students=100, max_questions_per_student=20):
        """Simulate training data from multiple students"""
        print("Generating training data...")
        
        for student_id in range(num_students):
            student_ability = np.random.normal(0.5, 0.2)  # Student ability between 0 and 1
            student_ability = max(0.1, min(0.9, student_ability))
            
            student_history = []
            current_difficulty = 1  # Start with medium
            
            for q_num in range(random.randint(5, max_questions_per_student)):
                # Select question based on current difficulty
                difficulty_names = ['Easy', 'Medium', 'Hard']
                available_questions = self.questions[difficulty_names[current_difficulty]]
                question = random.choice(available_questions)
                
                # Simulate response based on student ability and question difficulty
                difficulty_factor = (question.difficulty_level + 1) / 3.0
                success_prob = student_ability / difficulty_factor
                is_correct = random.random() < success_prob
                
                # Simulate response time (harder questions take longer, incorrect answers take longer)
                base_time = 10 + question.difficulty_level * 15
                time_variance = random.uniform(0.5, 2.0)
                if not is_correct:
                    time_variance *= 1.5  # Wrong answers take longer
                response_time = base_time * time_variance
                
                interaction = StudentInteraction(
                    question.id, question.difficulty_level, is_correct, response_time
                )
                student_history.append(interaction)
                
                # Update difficulty for next question
                if is_correct and response_time < base_time:
                    current_difficulty = min(2, current_difficulty + 1)  # Increase difficulty
                elif not is_correct or response_time > base_time * 1.5:
                    current_difficulty = max(0, current_difficulty - 1)  # Decrease difficulty
            
            self.interaction_history.extend(student_history)
    
    def _prepare_training_batch(self, batch_size=32, seq_len=10):
        """Prepare training batches from interaction history"""
        batches = []
        
        # Group interactions by students (simulate based on similar patterns)
        student_sequences = []
        current_seq = []
        
        for interaction in self.interaction_history:
            current_seq.append(interaction)
            if len(current_seq) >= seq_len + 1:  # +1 for target
                student_sequences.append(current_seq[:seq_len + 1])
                current_seq = current_seq[seq_len//2:]  # Sliding window
        
        # Create batches
        random.shuffle(student_sequences)
        for i in range(0, len(student_sequences), batch_size):
            batch_sequences = student_sequences[i:i + batch_size]
            if len(batch_sequences) < batch_size:
                continue
                
            # Prepare tensors
            question_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            difficulty_levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
            correctness = torch.zeros(batch_size, seq_len, dtype=torch.long)
            response_times = torch.zeros(batch_size, seq_len, dtype=torch.float)
            targets = torch.zeros(batch_size, dtype=torch.long)
            
            for j, sequence in enumerate(batch_sequences):
                for k in range(seq_len):
                    if k < len(sequence) - 1:
                        interaction = sequence[k]
                        question_ids[j, k] = interaction.question_id
                        difficulty_levels[j, k] = interaction.difficulty_level
                        correctness[j, k] = int(interaction.is_correct)
                        response_times[j, k] = min(interaction.response_time / 100.0, 2.0)  # Normalize
                
                # Target is the difficulty of the next question
                if len(sequence) > seq_len:
                    targets[j] = sequence[seq_len].difficulty_level
            
            batches.append((question_ids, difficulty_levels, correctness, response_times, targets))
        
        return batches
    
    def _train_model(self, epochs=50):
        """Train the transformer model"""
        print("Training transformer model...")
        self.model.train()
        
        training_batches = self._prepare_training_batch()
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_batches)
            
            for batch in training_batches:
                question_ids, difficulty_levels, correctness, response_times, targets = batch
                
                self.optimizer.zero_grad()
                
                difficulty_logits, confidence = self.model(
                    question_ids, difficulty_levels, correctness, response_times
                )
                
                loss = self.criterion(difficulty_logits, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(training_batches)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        self.model.eval()
    
    def predict_next_difficulty(self, student_history: List[StudentInteraction]) -> int:
        """Predict the next question difficulty based on student history"""
        if not student_history:
            return 1  # Start with medium difficulty
        
        self.model.eval()
        with torch.no_grad():
            # Prepare input tensors
            seq_len = min(len(student_history), 20)  # Use last 20 interactions
            recent_history = student_history[-seq_len:]
            
            question_ids = torch.zeros(1, seq_len, dtype=torch.long)
            difficulty_levels = torch.zeros(1, seq_len, dtype=torch.long)
            correctness = torch.zeros(1, seq_len, dtype=torch.long)
            response_times = torch.zeros(1, seq_len, dtype=torch.float)
            
            for i, interaction in enumerate(recent_history):
                question_ids[0, i] = interaction.question_id
                difficulty_levels[0, i] = interaction.difficulty_level
                correctness[0, i] = int(interaction.is_correct)
                response_times[0, i] = min(interaction.response_time / 100.0, 2.0)
            
            difficulty_logits, confidence = self.model(
                question_ids, difficulty_levels, correctness, response_times
            )
            
            predicted_difficulty = torch.argmax(difficulty_logits, dim=1).item()
            confidence_score = confidence.item()
            
            return predicted_difficulty, confidence_score
    
    def select_next_question(self, student_history: List[StudentInteraction]) -> Question:
        """Select the next question based on predicted difficulty"""
        if len(student_history) < 3:
            # For new students, start with easy questions
            difficulty_level = 0
        else:
            difficulty_level, confidence = self.predict_next_difficulty(student_history)
            
            # Add some randomness and rule-based adjustments
            recent_performance = [h.is_correct for h in student_history[-3:]]
            recent_times = [h.response_time for h in student_history[-3:]]
            
            # Rule-based adjustments
            if all(recent_performance) and all(t < 30 for t in recent_times):
                difficulty_level = min(2, difficulty_level + 1)  # Increase difficulty
            elif not any(recent_performance) or all(t > 60 for t in recent_times):
                difficulty_level = max(0, difficulty_level - 1)  # Decrease difficulty
        
        difficulty_names = ['Easy', 'Medium', 'Hard']
        available_questions = self.questions[difficulty_names[difficulty_level]]
        
        # Avoid repeating recent questions
        recent_question_ids = {h.question_id for h in student_history[-5:]}
        unused_questions = [q for q in available_questions if q.id not in recent_question_ids]
        
        if unused_questions:
            return random.choice(unused_questions)
        else:
            return random.choice(available_questions)
    
    def run_assessment(self):
        """Run the interactive assessment"""
        print("\n" + "="*60)
        print("ADAPTIVE ASSESSMENT SYSTEM")
        print("Powered by Transformer-based Question Selection")
        print("="*60)
        print("\nWelcome! This system will adapt question difficulty based on your performance.")
        print("Answer questions by typing the option number (1-4).")
        print("Type 'quit' to exit at any time.\n")
        
        student_history = []
        total_questions = 0
        correct_answers = 0
        
        while total_questions < 20:  # Maximum 20 questions
            # Select next question
            question = self.select_next_question(student_history)
            
            print(f"\n--- Question {total_questions + 1} ---")
            print(f"Difficulty: {question.difficulty}")
            print(f"Question: {question.text}")
            
            for i, option in enumerate(question.options, 1):
                print(f"{i}. {option}")
            
            # Record start time
            start_time = time.time()
            
            # Get user input
            while True:
                try:
                    user_input = input("\nYour answer (1-4): ").strip()
                    
                    if user_input.lower() == 'quit':
                        print("\nThanks for taking the assessment!")
                        self._show_final_results(student_history, correct_answers, total_questions)
                        return
                    
                    answer = int(user_input)
                    if 1 <= answer <= 4:
                        break
                    else:
                        print("Please enter a number between 1 and 4.")
                except ValueError:
                    print("Please enter a valid number or 'quit' to exit.")
            
            # Calculate response time
            response_time = time.time() - start_time
            is_correct = (answer - 1) == question.correct_answer
            
            # Record interaction
            interaction = StudentInteraction(
                question.id, question.difficulty_level, is_correct, response_time
            )
            student_history.append(interaction)
            
            # Provide feedback
            if is_correct:
                print("✓ Correct!")
                correct_answers += 1
            else:
                correct_option = question.options[question.correct_answer]
                print(f"✗ Incorrect. The correct answer was: {correct_option}")
            
            print(f"Time taken: {response_time:.1f} seconds")
            
            total_questions += 1
            
            # Show adaptation info
            if len(student_history) >= 2:
                try:
                    next_difficulty, confidence = self.predict_next_difficulty(student_history)
                    difficulty_names = ['Easy', 'Medium', 'Hard']
                    print(f"Next question will likely be: {difficulty_names[next_difficulty]} (confidence: {confidence:.2f})")
                except:
                    pass
        
        print("\nAssessment completed!")
        self._show_final_results(student_history, correct_answers, total_questions)
    
    def _show_final_results(self, student_history, correct_answers, total_questions):
        """Show final assessment results"""
        if total_questions == 0:
            return
            
        print("\n" + "="*50)
        print("ASSESSMENT RESULTS")
        print("="*50)
        
        accuracy = correct_answers / total_questions
        avg_time = sum(h.response_time for h in student_history) / len(student_history)
        
        difficulty_breakdown = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        difficulty_correct = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        
        for interaction in student_history:
            diff_name = ['Easy', 'Medium', 'Hard'][interaction.difficulty_level]
            difficulty_breakdown[diff_name] += 1
            if interaction.is_correct:
                difficulty_correct[diff_name] += 1
        
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Average Response Time: {avg_time:.1f} seconds")
        
        print("\nPerformance by Difficulty:")
        for diff in ['Easy', 'Medium', 'Hard']:
            if difficulty_breakdown[diff] > 0:
                diff_accuracy = difficulty_correct[diff] / difficulty_breakdown[diff]
                print(f"  {diff}: {difficulty_correct[diff]}/{difficulty_breakdown[diff]} ({diff_accuracy:.1%})")
        
        # Estimated ability level
        if accuracy > 0.8:
            ability_level = "Advanced"
        elif accuracy > 0.6:
            ability_level = "Intermediate"
        else:
            ability_level = "Beginner"
        
        print(f"\nEstimated Ability Level: {ability_level}")
        
        print("\nThank you for using the Adaptive Assessment System!")

def main():
    """Main function to run the assessment system"""
    print("Initializing Adaptive Assessment System...")
    print("This may take a moment as we train the AI model...")
    
    try:
        system = AdaptiveAssessmentSystem()
        system.run_assessment()
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please try again.")

if __name__ == "__main__":
    main()