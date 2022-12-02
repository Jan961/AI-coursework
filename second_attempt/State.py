

class State:
    def __init__(self, no_rectangles, Mond_score, is_valid):
        self.no_rectangles= no_rectangles
        self.Mond_score = Mond_score
        self.is_valid = is_valid

    def __str__(self):
        return f"No rectangles: {self.no_rectangles} \n" \
               f"Mondrian score: {self.Mond_score} \n"  \
               f"is valid: {self.is_valid}"