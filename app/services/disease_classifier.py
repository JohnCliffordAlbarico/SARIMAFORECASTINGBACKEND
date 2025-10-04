# app/services/disease_classifier.py
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiseaseCategory:
    """Enhanced disease category with medical taxonomy"""
    code: str
    name: str
    description: str
    keywords: List[str]
    icd10_codes: List[str]
    severity_indicators: List[str]
    seasonal_pattern: str
    typical_age_groups: List[str]
    contagious: bool
    chronic: bool

class EnhancedDiseaseClassifier:
    """
    Advanced disease classification system with medical taxonomy
    
    Features:
    - ICD-10 inspired categorization
    - Severity level detection
    - Seasonal pattern recognition
    - Age group correlation
    - Symptom-based classification
    """
    
    def __init__(self):
        self.disease_categories = self._initialize_disease_taxonomy()
        self.severity_keywords = self._initialize_severity_keywords()
        
    def _initialize_disease_taxonomy(self) -> Dict[str, DiseaseCategory]:
        """Initialize comprehensive disease taxonomy"""
        
        categories = {
            'respiratory_upper': DiseaseCategory(
                code='RESP_UPPER',
                name='Upper Respiratory Infections',
                description='Common cold, flu, sinusitis, pharyngitis',
                keywords=['cold', 'flu', 'cough', 'sore throat', 'runny nose', 'nasal congestion', 
                         'sinusitis', 'pharyngitis', 'laryngitis', 'rhinitis', 'sneezing'],
                icd10_codes=['J00-J06', 'J30-J39'],
                severity_indicators=['mild fever', 'low grade fever'],
                seasonal_pattern='winter_peak',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'respiratory_lower': DiseaseCategory(
                code='RESP_LOWER',
                name='Lower Respiratory Infections',
                description='Pneumonia, bronchitis, bronchiolitis',
                keywords=['pneumonia', 'bronchitis', 'bronchiolitis', 'chest infection', 
                         'lung infection', 'difficulty breathing', 'shortness of breath',
                         'chest pain', 'productive cough', 'wheezing'],
                icd10_codes=['J12-J18', 'J20-J22', 'J40-J47'],
                severity_indicators=['high fever', 'severe', 'hospitalization'],
                seasonal_pattern='winter_peak',
                typical_age_groups=['elderly', 'children', 'immunocompromised'],
                contagious=True,
                chronic=False
            ),
            
            'gastrointestinal_acute': DiseaseCategory(
                code='GI_ACUTE',
                name='Acute Gastrointestinal Disorders',
                description='Food poisoning, gastroenteritis, acute diarrhea',
                keywords=['diarrhea', 'vomiting', 'nausea', 'stomach pain', 'abdominal pain',
                         'food poisoning', 'gastroenteritis', 'stomach flu', 'dehydration',
                         'loose stools', 'stomach cramps'],
                icd10_codes=['K59', 'A08', 'A09', 'K30'],
                severity_indicators=['severe dehydration', 'blood in stool'],
                seasonal_pattern='summer_peak',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'gastrointestinal_chronic': DiseaseCategory(
                code='GI_CHRONIC',
                name='Chronic Gastrointestinal Disorders',
                description='IBS, IBD, GERD, chronic gastritis',
                keywords=['ibs', 'irritable bowel', 'crohn', 'ulcerative colitis', 'gerd',
                         'acid reflux', 'chronic gastritis', 'inflammatory bowel',
                         'chronic abdominal pain', 'chronic diarrhea'],
                icd10_codes=['K58', 'K50', 'K51', 'K21'],
                severity_indicators=['chronic', 'persistent', 'long-term'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'infectious_viral': DiseaseCategory(
                code='INF_VIRAL',
                name='Viral Infections',
                description='Viral fever, viral syndrome, common viral infections',
                keywords=['viral fever', 'viral infection', 'viral syndrome', 'fever',
                         'body aches', 'fatigue', 'malaise', 'headache', 'chills'],
                icd10_codes=['B34', 'R50'],
                severity_indicators=['high fever', 'severe fatigue'],
                seasonal_pattern='seasonal_variation',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'infectious_bacterial': DiseaseCategory(
                code='INF_BACTERIAL',
                name='Bacterial Infections',
                description='UTI, skin infections, bacterial pneumonia',
                keywords=['uti', 'urinary tract infection', 'skin infection', 'cellulitis',
                         'bacterial infection', 'abscess', 'wound infection', 'sepsis',
                         'bacterial pneumonia', 'strep throat'],
                icd10_codes=['N39', 'L03', 'A41', 'J15'],
                severity_indicators=['pus', 'severe', 'sepsis', 'high white count'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'cardiovascular': DiseaseCategory(
                code='CARDIO',
                name='Cardiovascular Disorders',
                description='Hypertension, heart disease, chest pain',
                keywords=['hypertension', 'high blood pressure', 'chest pain', 'heart disease',
                         'cardiac', 'palpitations', 'heart attack', 'angina', 'arrhythmia',
                         'heart failure', 'coronary'],
                icd10_codes=['I10-I15', 'I20-I25', 'I30-I52'],
                severity_indicators=['severe chest pain', 'heart attack', 'acute'],
                seasonal_pattern='winter_increase',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'endocrine_metabolic': DiseaseCategory(
                code='ENDO_META',
                name='Endocrine & Metabolic Disorders',
                description='Diabetes, thyroid disorders, metabolic syndrome',
                keywords=['diabetes', 'blood sugar', 'thyroid', 'hyperthyroid', 'hypothyroid',
                         'metabolic syndrome', 'obesity', 'weight gain', 'weight loss',
                         'fatigue', 'hormone', 'endocrine'],
                icd10_codes=['E10-E14', 'E00-E07', 'E88'],
                severity_indicators=['uncontrolled', 'severe', 'diabetic emergency'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'musculoskeletal_acute': DiseaseCategory(
                code='MSK_ACUTE',
                name='Acute Musculoskeletal Injuries',
                description='Sprains, strains, acute back pain, sports injuries',
                keywords=['sprain', 'strain', 'acute back pain', 'muscle pain', 'joint pain',
                         'sports injury', 'trauma', 'fracture', 'dislocation', 'acute pain',
                         'injury', 'accident'],
                icd10_codes=['S00-S99', 'M25', 'M79'],
                severity_indicators=['severe pain', 'unable to move', 'fracture'],
                seasonal_pattern='summer_sports',
                typical_age_groups=['young_adults', 'adults'],
                contagious=False,
                chronic=False
            ),
            
            'musculoskeletal_chronic': DiseaseCategory(
                code='MSK_CHRONIC',
                name='Chronic Musculoskeletal Disorders',
                description='Arthritis, chronic back pain, fibromyalgia',
                keywords=['arthritis', 'chronic back pain', 'chronic pain', 'fibromyalgia',
                         'osteoarthritis', 'rheumatoid arthritis', 'joint stiffness',
                         'chronic joint pain', 'degenerative', 'chronic neck pain'],
                icd10_codes=['M15-M19', 'M05-M14', 'M79.3'],
                severity_indicators=['chronic', 'persistent', 'severe disability'],
                seasonal_pattern='winter_worse',
                typical_age_groups=['elderly', 'adults'],
                contagious=False,
                chronic=True
            ),
            
            'neurological': DiseaseCategory(
                code='NEURO',
                name='Neurological Disorders',
                description='Headaches, migraines, seizures, neuropathy',
                keywords=['headache', 'migraine', 'seizure', 'epilepsy', 'neuropathy',
                         'nerve pain', 'numbness', 'tingling', 'dizziness', 'vertigo',
                         'neurological', 'brain', 'stroke'],
                icd10_codes=['G43', 'G40', 'G60-G64', 'G93'],
                severity_indicators=['severe headache', 'status epilepticus', 'stroke'],
                seasonal_pattern='weather_sensitive',
                typical_age_groups=['all_ages'],
                contagious=False,
                chronic=True
            ),
            
            'dermatological': DiseaseCategory(
                code='DERM',
                name='Dermatological Conditions',
                description='Skin rashes, eczema, dermatitis, skin infections',
                keywords=['rash', 'eczema', 'dermatitis', 'skin infection', 'itching',
                         'skin condition', 'allergic reaction', 'hives', 'psoriasis',
                         'acne', 'skin lesion', 'wound'],
                icd10_codes=['L20-L30', 'L40-L45', 'L50-L54'],
                severity_indicators=['severe reaction', 'widespread', 'infected'],
                seasonal_pattern='seasonal_allergies',
                typical_age_groups=['all_ages'],
                contagious=False,
                chronic=False
            ),
            
            'mental_health': DiseaseCategory(
                code='MENTAL',
                name='Mental Health Disorders',
                description='Anxiety, depression, stress-related disorders',
                keywords=['anxiety', 'depression', 'stress', 'panic', 'mental health',
                         'psychological', 'mood disorder', 'insomnia', 'sleep disorder',
                         'emotional', 'psychiatric'],
                icd10_codes=['F32-F33', 'F40-F48', 'F51'],
                severity_indicators=['severe', 'suicidal', 'psychotic'],
                seasonal_pattern='winter_depression',
                typical_age_groups=['adults', 'young_adults'],
                contagious=False,
                chronic=True
            ),
            
            'pediatric_specific': DiseaseCategory(
                code='PEDIATRIC',
                name='Pediatric-Specific Conditions',
                description='Childhood diseases, developmental issues',
                keywords=['childhood disease', 'pediatric', 'developmental delay',
                         'growth issues', 'vaccination', 'infant', 'child development',
                         'teething', 'colic', 'diaper rash'],
                icd10_codes=['P00-P96', 'Q00-Q99'],
                severity_indicators=['failure to thrive', 'severe developmental delay'],
                seasonal_pattern='school_year',
                typical_age_groups=['infants', 'children'],
                contagious=False,
                chronic=False
            ),
            
            'reproductive_health': DiseaseCategory(
                code='REPRO',
                name='Reproductive Health',
                description='Gynecological, urological, reproductive issues',
                keywords=['gynecological', 'menstrual', 'pregnancy', 'reproductive',
                         'urological', 'prostate', 'sexual health', 'fertility',
                         'contraception', 'pelvic pain'],
                icd10_codes=['N70-N98', 'N40-N53', 'O00-O99'],
                severity_indicators=['severe bleeding', 'emergency', 'complications'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults'],
                contagious=False,
                chronic=False
            )
        }
        
        return categories
    
    def _initialize_severity_keywords(self) -> Dict[str, List[str]]:
        """Initialize severity classification keywords"""
        
        return {
            'mild': ['mild', 'slight', 'minor', 'low grade', 'manageable', 'tolerable'],
            'moderate': ['moderate', 'significant', 'noticeable', 'concerning', 'persistent'],
            'severe': ['severe', 'intense', 'excruciating', 'unbearable', 'emergency', 
                      'acute', 'critical', 'hospitalization', 'admission'],
            'chronic': ['chronic', 'long-term', 'persistent', 'ongoing', 'recurrent',
                       'long-standing', 'years', 'months']
        }
    
    def classify_disease(self, chief_complaint: str, management: str = "", 
                        patient_age: Optional[int] = None) -> Dict[str, any]:
        """
        Classify disease based on chief complaint and management
        
        Returns:
            Dict containing category, confidence, severity, and additional metadata
        """
        
        if not chief_complaint:
            return self._default_classification()
        
        # Normalize text
        text = f"{chief_complaint} {management}".lower().strip()
        
        # Calculate scores for each category
        category_scores = {}
        for cat_code, category in self.disease_categories.items():
            score = self._calculate_category_score(text, category, patient_age)
            if score > 0:
                category_scores[cat_code] = score
        
        if not category_scores:
            return self._default_classification()
        
        # Get best match
        best_category_code = max(category_scores, key=category_scores.get)
        best_category = self.disease_categories[best_category_code]
        confidence = min(category_scores[best_category_code] / 10.0, 1.0)  # Normalize to 0-1
        
        # Determine severity
        severity = self._determine_severity(text)
        
        # Additional metadata
        metadata = {
            'seasonal_pattern': best_category.seasonal_pattern,
            'contagious': best_category.contagious,
            'chronic': best_category.chronic,
            'typical_age_groups': best_category.typical_age_groups,
            'icd10_codes': best_category.icd10_codes
        }
        
        return {
            'category_code': best_category_code,
            'category_name': best_category.name,
            'description': best_category.description,
            'confidence': confidence,
            'severity': severity,
            'metadata': metadata,
            'all_scores': category_scores
        }
    
    def _calculate_category_score(self, text: str, category: DiseaseCategory, 
                                 patient_age: Optional[int] = None) -> float:
        """Calculate matching score for a category"""
        
        score = 0.0
        
        # Keyword matching with weighted scoring
        for keyword in category.keywords:
            if keyword in text:
                # Longer keywords get higher scores
                weight = len(keyword.split()) * 2
                score += weight
        
        # Age group bonus
        if patient_age and self._age_matches_category(patient_age, category.typical_age_groups):
            score += 1.0
        
        # Severity indicator bonus
        for indicator in category.severity_indicators:
            if indicator in text:
                score += 1.5
        
        return score
    
    def _age_matches_category(self, age: int, age_groups: List[str]) -> bool:
        """Check if patient age matches category's typical age groups"""
        
        age_mapping = {
            'infants': (0, 2),
            'children': (2, 12),
            'young_adults': (13, 30),
            'adults': (18, 65),
            'elderly': (65, 120),
            'all_ages': (0, 120),
            'immunocompromised': (0, 120)  # Special category
        }
        
        for group in age_groups:
            if group in age_mapping:
                min_age, max_age = age_mapping[group]
                if min_age <= age <= max_age:
                    return True
        
        return False
    
    def _determine_severity(self, text: str) -> str:
        """Determine severity level from text"""
        
        severity_scores = {}
        
        for severity, keywords in self.severity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                severity_scores[severity] = score
        
        if not severity_scores:
            return 'mild'
        
        # Return highest scoring severity
        return max(severity_scores, key=severity_scores.get)
    
    def _default_classification(self) -> Dict[str, any]:
        """Return default classification for unmatched cases"""
        
        return {
            'category_code': 'UNCLASSIFIED',
            'category_name': 'Unclassified',
            'description': 'Unable to classify based on available information',
            'confidence': 0.0,
            'severity': 'mild',
            'metadata': {
                'seasonal_pattern': 'no_pattern',
                'contagious': False,
                'chronic': False,
                'typical_age_groups': ['all_ages'],
                'icd10_codes': []
            },
            'all_scores': {}
        }
    
    def get_category_statistics(self, classifications: List[Dict]) -> Dict[str, any]:
        """Generate statistics from a list of classifications"""
        
        if not classifications:
            return {}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        seasonal_patterns = {}
        
        for classification in classifications:
            # Category counts
            cat_code = classification.get('category_code', 'UNCLASSIFIED')
            category_counts[cat_code] = category_counts.get(cat_code, 0) + 1
            
            # Severity counts
            severity = classification.get('severity', 'mild')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Seasonal patterns
            pattern = classification.get('metadata', {}).get('seasonal_pattern', 'no_pattern')
            seasonal_patterns[pattern] = seasonal_patterns.get(pattern, 0) + 1
        
        return {
            'total_cases': len(classifications),
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'seasonal_patterns': seasonal_patterns,
            'average_confidence': sum(c.get('confidence', 0) for c in classifications) / len(classifications)
        }
    
    def get_all_categories(self) -> Dict[str, Dict]:
        """Get all available disease categories with metadata"""
        
        return {
            code: {
                'name': cat.name,
                'description': cat.description,
                'seasonal_pattern': cat.seasonal_pattern,
                'contagious': cat.contagious,
                'chronic': cat.chronic,
                'typical_age_groups': cat.typical_age_groups,
                'keyword_count': len(cat.keywords)
            }
            for code, cat in self.disease_categories.items()
        }
    
    def calculate_outbreak_probability(self, historical_data: pd.DataFrame, 
                                     disease_category: str, 
                                     forecast_period: int = 30,
                                     seasonal_adjustment: bool = True) -> Dict[str, Any]:
        """
        Calculate outbreak probability for a specific disease category
        
        Args:
            historical_data: DataFrame with historical disease cases
            disease_category: Disease category code
            forecast_period: Number of days to forecast
            seasonal_adjustment: Whether to apply seasonal adjustments
            
        Returns:
            Dict containing outbreak probability and related metrics
        """
        
        if disease_category not in self.disease_categories:
            return self._default_outbreak_prediction(disease_category)
        
        category = self.disease_categories[disease_category]
        
        # Filter data for this disease category
        category_data = self._filter_data_by_category(historical_data, disease_category)
        
        if category_data.empty or len(category_data) < 10:
            return self._default_outbreak_prediction(disease_category, low_data=True)
        
        # Calculate baseline statistics
        baseline_stats = self._calculate_baseline_statistics(category_data)
        
        # Calculate seasonal factors
        seasonal_factors = self._calculate_seasonal_factors(category_data, category.seasonal_pattern)
        
        # Calculate trend factors
        trend_factors = self._calculate_trend_factors(category_data)
        
        # Calculate outbreak probability
        outbreak_probability = self._calculate_probability_score(
            baseline_stats, seasonal_factors, trend_factors, 
            category, forecast_period, seasonal_adjustment
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(outbreak_probability)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            category_data, outbreak_probability, forecast_period
        )
        
        # Generate predictions by time period
        predicted_cases = self._generate_case_predictions(
            baseline_stats, seasonal_factors, trend_factors, forecast_period
        )
        
        return {
            'disease_category': disease_category,
            'category_name': category.name,
            'outbreak_probability': round(outbreak_probability, 2),
            'risk_level': risk_level,
            'confidence_level': self._calculate_confidence_level(category_data, category),
            'predicted_cases': predicted_cases,
            'historical_average': baseline_stats['average_cases'],
            'percentage_change': self._calculate_percentage_change(baseline_stats, predicted_cases),
            'seasonal_pattern': category.seasonal_pattern,
            'severity_distribution': self._estimate_severity_distribution(category_data, category),
            'age_group_risk': self._calculate_age_group_risk(category_data, category),
            'contagious': category.contagious,
            'chronic': category.chronic,
            'confidence_intervals': confidence_intervals,
            'peak_probability_date': self._estimate_peak_date(predicted_cases),
            'peak_cases_estimate': max(predicted_cases.values()) if predicted_cases else 0
        }
    
    def _filter_data_by_category(self, data: pd.DataFrame, category_code: str) -> pd.DataFrame:
        """Filter historical data by disease category"""
        
        if category_code not in self.disease_categories:
            return pd.DataFrame()
        
        category = self.disease_categories[category_code]
        
        # Create a mask for records matching this category
        mask = pd.Series([False] * len(data))
        
        for _, row in data.iterrows():
            chief_complaint = str(row.get('chief_complaint', '')).lower()
            management = str(row.get('management', '')).lower()
            text = f"{chief_complaint} {management}"
            
            # Check if any keywords match
            for keyword in category.keywords:
                if keyword in text:
                    mask.iloc[row.name] = True
                    break
        
        return data[mask]
    
    def _calculate_baseline_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline statistics from historical data"""
        
        if data.empty:
            return {'average_cases': 0, 'std_cases': 0, 'max_cases': 0, 'trend': 0}
        
        # Group by date and count cases
        if 'visit_date' in data.columns:
            daily_cases = data.groupby('visit_date').size()
        else:
            # Fallback to index-based grouping
            daily_cases = pd.Series([len(data) / 30] * 30)  # Approximate daily average
        
        return {
            'average_cases': float(daily_cases.mean()),
            'std_cases': float(daily_cases.std()) if len(daily_cases) > 1 else 0,
            'max_cases': float(daily_cases.max()),
            'trend': self._calculate_simple_trend(daily_cases)
        }
    
    def _calculate_seasonal_factors(self, data: pd.DataFrame, seasonal_pattern: str) -> Dict[str, float]:
        """Calculate seasonal adjustment factors"""
        
        from datetime import datetime
        current_month = datetime.now().month
        
        seasonal_multipliers = {
            'winter_peak': {12: 1.5, 1: 1.8, 2: 1.6, 3: 1.2, 4: 0.8, 5: 0.6, 
                           6: 0.5, 7: 0.5, 8: 0.6, 9: 0.8, 10: 1.0, 11: 1.3},
            'summer_peak': {12: 0.6, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.3, 
                           6: 1.6, 7: 1.8, 8: 1.5, 9: 1.2, 10: 0.8, 11: 0.6},
            'spring_peak': {12: 0.7, 1: 0.6, 2: 0.8, 3: 1.3, 4: 1.6, 5: 1.8, 
                           6: 1.2, 7: 0.8, 8: 0.6, 9: 0.8, 10: 1.0, 11: 0.8},
            'autumn_peak': {12: 1.0, 1: 0.6, 2: 0.5, 3: 0.6, 4: 0.8, 5: 1.0, 
                           6: 0.8, 7: 0.6, 8: 0.8, 9: 1.3, 10: 1.6, 11: 1.8},
            'no_pattern': {i: 1.0 for i in range(1, 13)},
            'seasonal_variation': {12: 1.2, 1: 1.3, 2: 1.1, 3: 0.9, 4: 0.8, 5: 0.7, 
                                  6: 0.8, 7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3}
        }
        
        multiplier_map = seasonal_multipliers.get(seasonal_pattern, seasonal_multipliers['no_pattern'])
        current_multiplier = multiplier_map.get(current_month, 1.0)
        
        return {
            'current_seasonal_factor': current_multiplier,
            'seasonal_pattern': seasonal_pattern,
            'peak_months': [k for k, v in multiplier_map.items() if v > 1.3]
        }
    
    def _calculate_trend_factors(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend factors from recent data"""
        
        if data.empty or len(data) < 7:
            return {'trend_direction': 0, 'trend_strength': 0, 'recent_increase': False}
        
        # Sort by date if available
        if 'visit_date' in data.columns:
            data_sorted = data.sort_values('visit_date')
            recent_data = data_sorted.tail(14)  # Last 2 weeks
            older_data = data_sorted.head(14)   # First 2 weeks
        else:
            # Fallback to simple split
            mid_point = len(data) // 2
            recent_data = data.iloc[mid_point:]
            older_data = data.iloc[:mid_point]
        
        recent_avg = len(recent_data) / 14 if len(recent_data) > 0 else 0
        older_avg = len(older_data) / 14 if len(older_data) > 0 else 0
        
        if older_avg > 0:
            trend_direction = (recent_avg - older_avg) / older_avg
        else:
            trend_direction = 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': abs(trend_direction),
            'recent_increase': trend_direction > 0.1
        }
    
    def _calculate_probability_score(self, baseline_stats: Dict, seasonal_factors: Dict, 
                                   trend_factors: Dict, category: DiseaseCategory, 
                                   forecast_period: int, seasonal_adjustment: bool) -> float:
        """Calculate overall outbreak probability score (0-100)"""
        
        # Base probability from historical average
        base_prob = min(baseline_stats['average_cases'] * 2, 50)  # Cap at 50%
        
        # Seasonal adjustment
        if seasonal_adjustment:
            seasonal_multiplier = seasonal_factors['current_seasonal_factor']
            base_prob *= seasonal_multiplier
        
        # Trend adjustment
        if trend_factors['recent_increase']:
            base_prob *= (1 + trend_factors['trend_strength'])
        
        # Contagious disease bonus
        if category.contagious:
            base_prob *= 1.2
        
        # Forecast period adjustment (longer periods = higher probability)
        period_multiplier = 1 + (forecast_period - 30) / 100
        base_prob *= period_multiplier
        
        # Ensure probability is between 0 and 100
        return max(0, min(100, base_prob))
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on outbreak probability"""
        
        if probability >= 75:
            return "Critical"
        elif probability >= 50:
            return "High"
        elif probability >= 25:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_confidence_level(self, data: pd.DataFrame, category: DiseaseCategory) -> float:
        """Calculate confidence level in the prediction"""
        
        # Base confidence on data quantity and quality
        data_points = len(data)
        
        if data_points >= 100:
            base_confidence = 90
        elif data_points >= 50:
            base_confidence = 80
        elif data_points >= 20:
            base_confidence = 70
        else:
            base_confidence = 60
        
        # Adjust for seasonal patterns (more predictable = higher confidence)
        if category.seasonal_pattern in ['winter_peak', 'summer_peak']:
            base_confidence += 5
        elif category.seasonal_pattern == 'no_pattern':
            base_confidence -= 5
        
        return max(50, min(95, base_confidence))
    
    def _calculate_confidence_intervals(self, data: pd.DataFrame, probability: float, 
                                     forecast_period: int) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for predictions"""
        
        # Simple confidence interval calculation
        margin = probability * 0.15  # 15% margin
        
        return {
            'outbreak_probability': {
                'lower': max(0, probability - margin),
                'upper': min(100, probability + margin)
            },
            'cases': {
                'lower': max(0, probability * 0.8),
                'upper': probability * 1.2
            }
        }
    
    def _generate_case_predictions(self, baseline_stats: Dict, seasonal_factors: Dict, 
                                 trend_factors: Dict, forecast_period: int) -> Dict[str, float]:
        """Generate predicted cases by time period"""
        
        base_daily_cases = baseline_stats['average_cases']
        seasonal_factor = seasonal_factors['current_seasonal_factor']
        trend_factor = 1 + trend_factors['trend_direction']
        
        predictions = {}
        
        # Weekly predictions
        for week in range(1, min(forecast_period // 7 + 1, 5)):
            weekly_cases = base_daily_cases * 7 * seasonal_factor * trend_factor
            predictions[f'week_{week}'] = round(weekly_cases, 1)
        
        # Monthly prediction if forecast period is long enough
        if forecast_period >= 30:
            monthly_cases = base_daily_cases * 30 * seasonal_factor * trend_factor
            predictions['month_1'] = round(monthly_cases, 1)
        
        return predictions
    
    def _calculate_percentage_change(self, baseline_stats: Dict, predicted_cases: Dict) -> float:
        """Calculate percentage change from historical average"""
        
        if not predicted_cases or baseline_stats['average_cases'] == 0:
            return 0
        
        # Use first week prediction for comparison
        first_week_pred = predicted_cases.get('week_1', baseline_stats['average_cases'] * 7)
        historical_weekly = baseline_stats['average_cases'] * 7
        
        if historical_weekly > 0:
            return round(((first_week_pred - historical_weekly) / historical_weekly) * 100, 1)
        
        return 0
    
    def _estimate_severity_distribution(self, data: pd.DataFrame, category: DiseaseCategory) -> Dict[str, float]:
        """Estimate severity distribution for the disease category"""
        
        # Default distributions based on disease type
        if category.chronic:
            return {'mild': 40.0, 'moderate': 45.0, 'severe': 15.0}
        elif category.contagious:
            return {'mild': 60.0, 'moderate': 30.0, 'severe': 10.0}
        else:
            return {'mild': 50.0, 'moderate': 35.0, 'severe': 15.0}
    
    def _calculate_age_group_risk(self, data: pd.DataFrame, category: DiseaseCategory) -> Dict[str, float]:
        """Calculate risk by age group"""
        
        # Default risk based on category's typical age groups
        base_risk = {'infants': 20.0, 'children': 25.0, 'young_adults': 30.0, 'adults': 35.0, 'elderly': 40.0}
        
        # Adjust based on category's typical age groups
        for age_group in category.typical_age_groups:
            if age_group in base_risk:
                base_risk[age_group] *= 1.5
        
        # Normalize to ensure total doesn't exceed reasonable bounds
        total_risk = sum(base_risk.values())
        if total_risk > 150:
            factor = 150 / total_risk
            base_risk = {k: v * factor for k, v in base_risk.items()}
        
        return {k: round(v, 1) for k, v in base_risk.items()}
    
    def _estimate_peak_date(self, predicted_cases: Dict) -> Optional[str]:
        """Estimate the most likely peak date"""
        
        if not predicted_cases:
            return None
        
        # Find the period with highest predicted cases
        max_cases = 0
        peak_period = None
        
        for period, cases in predicted_cases.items():
            if cases > max_cases:
                max_cases = cases
                peak_period = period
        
        # Convert period to approximate date
        from datetime import datetime, timedelta
        
        if peak_period and 'week' in peak_period:
            week_num = int(peak_period.split('_')[1])
            peak_date = datetime.now() + timedelta(weeks=week_num)
            return peak_date.strftime('%Y-%m-%d')
        elif peak_period and 'month' in peak_period:
            peak_date = datetime.now() + timedelta(days=30)
            return peak_date.strftime('%Y-%m-%d')
        
        return None
    
    def _calculate_simple_trend(self, series: pd.Series) -> float:
        """Calculate simple trend from time series"""
        
        if len(series) < 2:
            return 0
        
        # Simple linear trend calculation
        x = range(len(series))
        y = series.values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _default_outbreak_prediction(self, disease_category: str, low_data: bool = False) -> Dict[str, Any]:
        """Return default prediction when insufficient data"""
        
        category_name = self.disease_categories.get(disease_category, {}).get('name', 'Unknown Disease')
        
        return {
            'disease_category': disease_category,
            'category_name': category_name,
            'outbreak_probability': 15.0 if low_data else 5.0,
            'risk_level': 'Low',
            'confidence_level': 50.0,
            'predicted_cases': {'week_1': 1.0, 'week_2': 1.0},
            'historical_average': 0.5,
            'percentage_change': 0.0,
            'seasonal_pattern': 'no_pattern',
            'severity_distribution': {'mild': 70.0, 'moderate': 25.0, 'severe': 5.0},
            'age_group_risk': {'infants': 15.0, 'children': 20.0, 'young_adults': 25.0, 'adults': 25.0, 'elderly': 30.0},
            'contagious': False,
            'chronic': False,
            'confidence_intervals': {
                'outbreak_probability': {'lower': 0.0, 'upper': 25.0},
                'cases': {'lower': 0.0, 'upper': 2.0}
            },
            'peak_probability_date': None,
            'peak_cases_estimate': 1.0
        }

# Global instance
disease_classifier = EnhancedDiseaseClassifier()
