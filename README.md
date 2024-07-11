# Explainable Movie Recommendation System

This project implements an explainable movie recommendation system using Model-Driven Engineering (MDE) techniques. It demonstrates how to integrate explainability into AI systems from the design phase, addressing the limitations of post-hoc explanation methods.

## Table of Contents
1. [Architecture](#architecture)
2. [Metamodel](#metamodel)
3. [Implementation Details](#implementation-details)
4. [Key Components](#key-components)
5. [Usage](#usage)
6. [Example Output](#example-output)
7. [Future Work](#future-work)

## Architecture

The system follows a Model-Driven Engineering approach with the following components:

1. Metamodel Definition: Defined in 'content_recommendation.ecore'
2. Model-to-Model Transformations: Implemented in the `transform_model` function
3. Code Generation: Handled by pyecore and pyecoregen
4. Domain-Specific AI Models: Implemented in `OriginalAIModel`
5. Explainable AI Model: Implemented in `ExplainableAIModel`
6. Explanation Tracing & Logging: Implemented in `AIModelTracer`
7. Explanation Generation Strategies: Implemented in `predictAndExplain` method of `ExplainableAIModel`
8. Real-time Explanation Integration: Implemented in `RealTimeExplainer`
9. Explanation Quality Assessment: Implemented in `ExplanationQualityAssessor`

## Metamodel

The system is based on an Ecore metamodel that defines the structure of the recommendation system. Key elements include:
![metamodel](https://github.com/MA-ysr/MDE_XAI_MODELS2024/assets/174721531/2a2d90dc-f451-4336-832e-928ccb99246f)

**User:** Represents a user of the recommendation system.
- Attributes: id (EInt), name (EString)
- Role: Represents the entity receiving recommendations.

**Content:** Represents an item that can be recommended (e.g., a movie).
- Attributes: id (EInt), title (EString), genre (EString)
- Role: Represents the items available for recommendation.

**Rating:** Represents a user's rating for a specific content item.
- References: user (User), content (Content)
- Attribute: score (EFloat)
- Role: Captures user preferences, which are used to train the AI model.

**Recommendation:** Represents a recommendation made by the system.
- References: user (User), recommendedContent (Content), explanation (Explanation)
- Attribute: predictedRating (EFloat)
- Role: Encapsulates the system's output, linking a user to recommended content with a predicted rating and explanation.

**Explanation:** Represents the explanation for a recommendation.
- Attribute: type (ExplanationType), content (EString)
- References: factors (ExplanationFactor, multiple)
- Role: Provides interpretable reasons for a recommendation.

**ExplanationFactor:** Represents a specific factor contributing to an explanation.
- Attributes: name (EString), value (EFloat), importance (EFloat, derived)
- Role: Breaks down the explanation into individual, quantifiable factors.

**AIRecommendationEngine:** The main class orchestrating the recommendation process.
- References: users (User, multiple), contentCatalog (Content, multiple), ratings (Rating, multiple), aiModel (AIModel), tracer (AIModelTracer)
- Operations: predictAndExplain, trainModel
- Role: Manages the overall recommendation system, including users, content, and the AI model.

**AIModel:** Represents the core AI model used for making predictions.
- Attribute: isExplainable (EBoolean)
- Operations: fit, predict, extractFactors
- Role: Encapsulates the AI algorithm used for generating recommendations.

**AIModelTracer:** Responsible for tracing predictions for analysis and debugging.
- Reference: traces (PredictionTrace, multiple)
- Operation: tracePrediction
- Role: Provides a mechanism for logging and analyzing the system's predictions.

**PredictionTrace:** Represents a logged prediction for analysis.
- Attributes: userId (EInt), contentId (EInt), predictedRating (EFloat)
- Reference: explanation (Explanation)
- Role: Stores individual prediction results for later analysis.

**ExplanationType (Enumeration):** Defines different types of explanations (FACTOR_BASED, CONTENT_BASED, COLLABORATIVE).
- Role: Allows for categorization of different explanation strategies.

**Relationships:**
* User to Rating: One-to-many (a user can have multiple ratings)
* Content to Rating: One-to-many (a content item can have multiple ratings)
* User to Recommendation: One-to-many (a user can receive multiple recommendations)
* Content to Recommendation: One-to-many (a content item can be recommended multiple times)
* Recommendation to Explanation: One-to-one (each recommendation has one explanation)
* Explanation to ExplanationFactor: One-to-many (an explanation consists of multiple factors)
* AIRecommendationEngine to AIModel: One-to-one (the engine uses one AI model)
* AIRecommendationEngine to AIModelTracer: One-to-one (the engine uses one tracer)
* AIModelTracer to PredictionTrace: One-to-many (the tracer logs multiple prediction traces)

This metamodel design encapsulates the key components of an explainable recommendation system, from the basic entities (User, Content) to the AI model and explanation mechanisms. It provides a structure that inherently supports the generation of explanations alongside recommendations, addressing the post-hoc limitation by making explainability an integral part of the system's design.

## Implementation Details

### 1. Imports and Setup
These imports set up the necessary libraries for matrix operations, machine learning, and model-driven engineering.
```python
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from pyecore.resources import ResourceSet, URI
from pyecore.ecore import EPackage
from pyecore.resources.xmi import XMIResource
from pyecore.ecore import EFloat
from functools import partial
```
### 2. Metamodel Loading and Class Generation
This code loads the Ecore metamodel and generates Python classes from it, forming the basis of our MDE approach.
```python
rset = ResourceSet()
resource = rset.get_resource(URI('content_recommendation.ecore'))
mm_root = resource.contents[0]
rset.metamodel_registry[mm_root.nsURI] = mm_root

from pyecoregen.ecore import EcoreGenerator 
generator = EcoreGenerator()
generator.generate(mm_root, 'content_recommendation_mm')
```
### 3. Importing generated classes
Here we import the classes generated from our metamodel, which will be used throughout the implementation.
```python
from content_recommendation_mm.airecommendationsystem import (
    User, Content, Rating, Recommendation, Explanation, ExplanationFactor, 
    AIRecommendationEngine, AIModel, AIModelTracer, PredictionTrace, ExplanationType,
    getEClassifier, eClassifiers
)
import content_recommendation_mm.airecommendationsystem as mm
```

## Key components
### 1. OriginalAIModel
This class implements the core recommendation algorithm using Non-Negative Matrix Factorization (NMF). It fits the model to the rating matrix and provides methods for prediction and factor extraction.
```python
class OriginalAIModel(AIModel):
    def __init__(self):
        super().__init__()
        self.isExplainable = False
        self.model = NMF(n_components=2, init='random', random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.scaler = MinMaxScaler(feature_range=(1.0, 5.0))

    def fit(self, rating_matrix):
        self.user_factors = self.model.fit_transform(rating_matrix)
        self.item_factors = self.model.components_.T
        
        all_predictions = np.dot(self.user_factors, self.item_factors.T).flatten()
        self.scaler.fit(all_predictions.reshape(-1, 1))

    def predict(self, user_idx, content_idx):
        user_vector = self.user_factors[user_idx]
        content_vector = self.item_factors[content_idx]
        predicted_rating = float(np.dot(user_vector, content_vector))
        return float(self.scaler.transform([[predicted_rating]])[0][0])

    def extractFactors(self, user_idx, content_idx):
        return self.user_factors[user_idx].astype(float), self.item_factors[content_idx].astype(float)
```
### 2. ExplainableAIModel
This class extends the original model with explainability features. It generates explanations based on the latent factors learned by the NMF model.
```python
class ExplainableAIModel(AIModel):
    def __init__(self, original_model):
        super().__init__()
        self.isExplainable = True
        self.original_model = original_model

    def fit(self, rating_matrix):
        self.original_model.fit(rating_matrix)

    def predict(self, user_idx, content_idx):
        return self.original_model.predict(user_idx, content_idx)

    def extractFactors(self, user_idx, content_idx):
        return self.original_model.extractFactors(user_idx, content_idx)

    def predictAndExplain(self, user_idx, content_idx):
        predicted_rating = self.predict(user_idx, content_idx)
        user_vector, content_vector = self.extractFactors(user_idx, content_idx)
        
        explanation = mm.Explanation()
        explanation.type = mm.ExplanationType.FACTOR_BASED
        explanation.content = f"The predicted rating of {predicted_rating:.2f} is based on the following factors:"

        factor_values = [abs(float(u*c)) for u, c in zip(user_vector, content_vector)]
        total_importance = sum(factor_values)
        
        for i, (user_factor, content_factor) in enumerate(zip(user_vector, content_vector)):
            factor = ExplanationFactorImpl()
            factor.name = f"Latent Factor {i+1}"
            factor.value = float(user_factor * content_factor)
            factor.importance = float(abs(factor.value) / total_importance) if total_importance != 0 else 0.0
            explanation.factors.append(factor)

        return predicted_rating, explanation
```

### 3. RealTimeExplainer
This class ensures that explanations are generated in real-time alongside predictions, addressing the post-hoc limitation of traditional XAI approaches.
```python
class RealTimeExplainer:
    def __init__(self, ai_model):
        self.ai_model = ai_model
        self.explanation_history = []

    def explain_in_realtime(self, user_idx, content_idx):
        predicted_rating, explanation = self.ai_model.predictAndExplain(user_idx, content_idx)
        self.explanation_history.append(explanation)
        return predicted_rating, explanation
```

### 4. AIRecommendationEngineImpl
This class orchestrates the entire recommendation process, including model training, prediction, and explanation generation.
```python
class AIRecommendationEngineImpl(AIRecommendationEngine):
    def __init__(self):
        super().__init__()
        self.original_model = OriginalAIModel()
        self.explainable_model = None
        self.real_time_explainer = None
        self.tracer = mm.AIModelTracer()
        self.user_id_map = {}
        self.content_id_map = {}

    def trainModel(self):
        self.user_id_map = {user.id: i for i, user in enumerate(self.users)}
        self.content_id_map = {content.id: i for i, content in enumerate(self.contentCatalog)}

        rating_matrix = np.zeros((len(self.users), len(self.contentCatalog)))
        for rating in self.ratings:
            user_idx = self.user_id_map[rating.user.id]
            content_idx = self.content_id_map[rating.content.id]
            rating_matrix[user_idx, content_idx] = rating.score

        rating_matrix += 0.01  # Add small constant to avoid zero entries
        self.original_model.fit(rating_matrix)
        self.explainable_model = transform_model(self.original_model)
        self.real_time_explainer = RealTimeExplainer(self.explainable_model)

    def predictAndExplain(self, user, content):
        user_idx = self.user_id_map[user.id]
        content_idx = self.content_id_map[content.id]

        predicted_rating, explanation = self.real_time_explainer.explain_in_realtime(user_idx, content_idx)
        
        recommendation = mm.Recommendation()
        recommendation.user = user
        recommendation.recommendedContent = content
        recommendation.predictedRating = float(predicted_rating)
        recommendation.explanation = explanation

        self.tracer.tracePrediction(user, content, recommendation)

        return recommendation, explanation

    # Additional methods for adding users, content, and ratings...
```
### 5. ExplanationQualityAssessor
This class provides a mechanism to assess the quality of generated explanations. It considers three aspects:
* Completeness: How many factors are included in the explanation.
* Coherence: Whether the importance values of factors are within a valid range.
* Relevance: How close the explanation's implied rating is to the actual rating (if provided).

The class keeps track of quality scores and can provide a summary of the overall explanation quality.
```python
class ExplanationQualityAssessor:
    def __init__(self):
        self.quality_scores = []

    def assess_explanation(self, explanation, actual_rating=None):
        # Assess completeness
        completeness = len(explanation.factors) / 2  # Assuming 2 factors is complete

        # Assess coherence
        coherence = 1.0 if all(f.importance >= 0 and f.importance <= 1 for f in explanation.factors) else 0.0

        # Assess relevance (if actual rating is provided)
        relevance = 1.0
        if actual_rating is not None:
            predicted_rating = sum(f.value for f in explanation.factors)
            relevance = 1 - min(1, abs(actual_rating - predicted_rating) / 4)  # Normalize to [0, 1]

        # Calculate overall quality score
        quality_score = (completeness + coherence + relevance) / 3
        self.quality_scores.append(quality_score)

        return quality_score

    def get_average_quality(self):
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0

    def get_quality_summary(self):
        return {
            'average_quality': self.get_average_quality(),
            'num_assessments': len(self.quality_scores),
            'min_quality': min(self.quality_scores) if self.quality_scores else 0,
            'max_quality': max(self.quality_scores) if self.quality_scores else 0
        }
```

### 6. Utility functions
These utility functions handle model transformation, prediction tracing, and user-friendly explanation generation.
```python
def transform_model(original_model):
    return ExplainableAIModel(original_model)

def trace_prediction(self, user, content, recommendation):
    trace = mm.PredictionTrace()
    trace.userId = user.id
    trace.contentId = content.id
    trace.predictedRating = recommendation.predictedRating
    trace.explanation = recommendation.explanation
    self.traces.append(trace)
    return trace

mm.AIModelTracer.tracePrediction = trace_prediction

def generate_user_friendly_explanation(user, content, predicted_rating, explanation):
    # ... (explanation generation logic)
    return explanation_text
```

## Usage
This code demonstrates how to create an instance of the recommendation engine, populate it with sample data, train the model, and generate recommendations with explanations.
```python
# Create and populate the AI Recommendation Engine
engine = AIRecommendationEngineImpl()

# Add users
for i in range(1, 6):
    engine.add_user(i, f"User{i}")

# Add content
content_data = [
    (1, "Action Movie 1", "Action"),
    (2, "Comedy Movie 1", "Comedy"),
    (3, "Drama Movie 1", "Drama"),
    (4, "Action Movie 2", "Action"),
    (5, "Comedy Movie 2", "Comedy")
]

for id, title, genre in content_data:
    engine.add_content(id, title, genre)

# Add ratings
for user in engine.users:
    for content in engine.contentCatalog:
        if np.random.random() > 0.2:  # 80% chance of rating each item
            score = np.random.uniform(1, 5)  # Generate a float between 1 and 5
            engine.add_rating(user, content, score)

# Train the model
engine.trainModel()

# Generate and print recommendations with explanations for each user
for user in engine.users:
    print(f"\nRecommendations for {user.name}:")
    for content in engine.contentCatalog:
        recommendation, explanation = engine.predictAndExplain(user, content)
        user_friendly_explanation = generate_user_friendly_explanation(user, content, recommendation.predictedRating, explanation)
        print(user_friendly_explanation)

# Print tracing information
print("\nPrediction Traces:")
for trace in engine.tracer.traces:
    print(f"User {trace.userId} - Content {trace.contentId}: Predicted Rating {trace.predictedRating:.2f}")
    if trace.explanation and trace.explanation.factors:
        print(f"  Explanation: {trace.explanation.content}")
        for factor in trace.explanation.factors:
            print(f"    - {factor.name}: Value = {factor.value:.2f}, Importance = {factor.importance:.2f}")
    else:
        print("  No detailed explanation available.")
```

## Example Output
The system provides personalized movie recommendations with explanations:
```
Recommendations for User1:
We predict you'll rate 'Action Movie 1' 3.8 out of 5 stars.
This is based on a combination of factors:
- It matches your general movie preferences (Impact: 70%)
- It's similar to other movies you've enjoyed (Impact: 30%)

Prediction Traces:
User 1 - Content 1: Predicted Rating 3.80
  Explanation: The predicted rating of 3.80 is based on the following factors:
    - Latent Factor 1: Value = 2.66, Importance = 0.70
    - Latent Factor 2: Value = 1.14, Importance = 0.30
```

## Future Work
* Implement more sophisticated explanation quality assessment
* Extend the metamodel to capture more domain-specific concepts
* Develop more advanced explanation generation strategies
* Integrate user feedback to improve explanation quality
* Explore different AI models and their impact on explainability
