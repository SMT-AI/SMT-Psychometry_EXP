import numpy as np

class MAUTCareerClassifier:
    def __init__(self):
        # Define career options
        self.careers = ['Software Engineer', 'Hardware Engineer', 'Costume Designer']
        
        # Define attributes (skills/interests measured by questions)
        self.attributes = [
            'Programming Logic',
            'Mathematical Reasoning', 
            'Hardware Understanding',
            'Circuit Analysis',
            'Creative Design',
            'Visual Aesthetics',
            'Problem Solving',
            'Technical Documentation',
            'Artistic Expression',
            'Detail Orientation'
        ]
        
        # Define questions (each maps to an attribute)
        self.questions = [
            "Rate your interest in solving complex programming puzzles (1-5): ",
            "How comfortable are you with mathematical calculations and formulas (1-5): ",
            "Rate your interest in understanding how electronic devices work internally (1-5): ",
            "How much do you enjoy analyzing electrical circuits and components (1-5): ",
            "Rate your passion for creating visually appealing designs (1-5): ",
            "How important is color coordination and visual harmony to you (1-5): ",
            "Rate your enjoyment in breaking down complex problems into smaller parts (1-5): ",
            "How comfortable are you with writing detailed technical explanations (1-5): ",
            "Rate your interest in expressing ideas through visual or artistic mediums (1-5): ",
            "How important is precision and attention to small details in your work (1-5): "
        ]
        
        # Career-attribute weight matrix (how important each attribute is for each career)
        # Rows: careers, Columns: attributes
        self.career_weights = np.array([
            # Prog, Math, Hard, Circ, Crea, Aest, Prob, Tech, Art, Detail
            [0.25, 0.20, 0.05, 0.05, 0.05, 0.05, 0.20, 0.15, 0.00, 0.10],  # Software Engineer
            [0.15, 0.25, 0.25, 0.20, 0.00, 0.00, 0.15, 0.15, 0.00, 0.20],  # Hardware Engineer  
            [0.00, 0.05, 0.00, 0.00, 0.30, 0.25, 0.10, 0.05, 0.30, 0.15]   # Costume Designer
        ])
        
        # Ensure weights sum to 1 for each career
        self.career_weights = self.career_weights / self.career_weights.sum(axis=1, keepdims=True)
        
    def normalize_scores(self, scores):
        """Normalize scores to 0-1 range"""
        return (np.array(scores) - 1) / 4  # Convert 1-5 scale to 0-1 scale
        
    def calculate_utility(self, user_scores):
        """Calculate utility scores for each career using MAUT"""
        normalized_scores = self.normalize_scores(user_scores)
        
        # Calculate weighted utility for each career
        utility_scores = np.zeros(len(self.careers))
        
        for i, career in enumerate(self.careers):
            # Weighted sum of normalized scores
            utility_scores[i] = np.sum(self.career_weights[i] * normalized_scores)
            
        return utility_scores
    
    def get_user_input(self):
        """Collect user responses to all questions"""
        print("=== CAREER PATH ASSESSMENT ===")
        print("Please answer each question on a scale of 1-5:")
        print("1 = Strongly Disagree/Not Interested")
        print("2 = Disagree/Slightly Interested") 
        print("3 = Neutral")
        print("4 = Agree/Interested")
        print("5 = Strongly Agree/Very Interested")
        print("-" * 50)
        
        user_scores = []
        
        for i, question in enumerate(self.questions):
            while True:
                try:
                    score = int(input(f"Q{i+1}: {question}"))
                    if 1 <= score <= 5:
                        user_scores.append(score)
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
        
        return user_scores
    
    def display_results(self, utility_scores, user_scores):
        """Display detailed results and recommendations"""
        print("\n" + "="*60)
        print("CAREER ASSESSMENT RESULTS")
        print("="*60)
        
        # Sort careers by utility score
        career_ranking = sorted(zip(self.careers, utility_scores), 
                              key=lambda x: x[1], reverse=True)
        
        print("\nCAREER MATCH RANKING:")
        print("-"*30)
        for i, (career, score) in enumerate(career_ranking):
            percentage = score * 100
            print(f"{i+1}. {career}: {percentage:.1f}% match")
        
        print(f"\nRECOMMENDED CAREER PATH: {career_ranking[0][0]}")
        
        # Show attribute breakdown for top career
        top_career_idx = self.careers.index(career_ranking[0][0])
        normalized_scores = self.normalize_scores(user_scores)
        
        print(f"\nATTRIBUTE ANALYSIS FOR {career_ranking[0][0].upper()}:")
        print("-"*50)
        
        for i, attr in enumerate(self.attributes):
            weight = self.career_weights[top_career_idx][i]
            user_score = normalized_scores[i]
            contribution = weight * user_score
            
            if weight > 0.1:  # Only show important attributes
                print(f"{attr}: {user_score*100:.0f}% score, "
                      f"{weight*100:.0f}% importance → "
                      f"{contribution*100:.1f}% contribution")
        
        # Career insights
        print(f"\nCARESR INSIGHTS:")
        print("-"*20)
        self.provide_career_insights(career_ranking[0][0], normalized_scores, top_career_idx)
    
    def provide_career_insights(self, career, scores, career_idx):
        """Provide specific insights based on the recommended career"""
        weights = self.career_weights[career_idx]
        
        if career == "Software Engineer":
            prog_score = scores[0]
            math_score = scores[1] 
            prob_score = scores[6]
            
            print(f"• Programming aptitude: {'Strong' if prog_score > 0.6 else 'Developing'}")
            print(f"• Mathematical foundation: {'Solid' if math_score > 0.6 else 'Needs improvement'}")
            print(f"• Problem-solving skills: {'Excellent' if prob_score > 0.7 else 'Good' if prob_score > 0.5 else 'Developing'}")
            
            if prog_score < 0.5:
                print("→ Consider taking programming courses to strengthen coding skills")
            if math_score < 0.5:
                print("→ Focus on mathematical concepts and logical reasoning")
                
        elif career == "Hardware Engineer":
            hard_score = scores[2]
            circ_score = scores[3]
            math_score = scores[1]
            
            print(f"• Hardware knowledge: {'Strong' if hard_score > 0.6 else 'Developing'}")
            print(f"• Circuit analysis: {'Proficient' if circ_score > 0.6 else 'Learning'}")
            print(f"• Mathematical skills: {'Strong' if math_score > 0.6 else 'Needs work'}")
            
            if hard_score < 0.5:
                print("→ Explore electronics projects and hardware fundamentals")
            if circ_score < 0.5:
                print("→ Study circuit analysis and electronic components")
                
        elif career == "Costume Designer":
            crea_score = scores[4]
            aest_score = scores[5]
            art_score = scores[8]
            
            print(f"• Creative design: {'Exceptional' if crea_score > 0.7 else 'Good' if crea_score > 0.5 else 'Developing'}")
            print(f"• Aesthetic sense: {'Refined' if aest_score > 0.6 else 'Developing'}")
            print(f"• Artistic expression: {'Strong' if art_score > 0.6 else 'Emerging'}")
            
            if crea_score < 0.5:
                print("→ Practice design projects and creative exercises")
            if art_score < 0.5:
                print("→ Explore various artistic mediums and techniques")

    def run_assessment(self):
        """Run the complete MAUT assessment"""
        user_scores = self.get_user_input()
        utility_scores = self.calculate_utility(user_scores)
        self.display_results(utility_scores, user_scores)
        
        return utility_scores, user_scores

# Main execution
if __name__ == "__main__":
    classifier = MAUTCareerClassifier()
    
    print("Welcome to the MAUT Career Path Classifier!")
    print("This assessment will help identify your best career match.")
    
    while True:
        utility_scores, user_scores = classifier.run_assessment()
        
        print("\n" + "="*60)
        retry = input("Would you like to take the assessment again? (y/n): ").lower()
        if retry != 'y':
            print("Thank you for using the Career Assessment Tool!")
            break
        print("\n")