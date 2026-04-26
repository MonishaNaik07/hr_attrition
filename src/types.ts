export interface User {
  username: string;
  password: string;
}

export interface PredictionResult {
  attritionProbability: number;
  riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
  riskFactors: string[];
  retentionActions: string[];
  pipeline: 'ibm' | 'synthetic';
}

export interface PredictionRecord {
  id: string;
  timestamp: string;
  type: 'individual' | 'batch';
  pipeline: 'ibm' | 'synthetic';
  input: Record<string, string | number>;
  result: PredictionResult;
  batchId?: string;
}

export interface BatchSummary {
  id: string;
  timestamp: string;
  pipeline: 'ibm' | 'synthetic';
  totalRecords: number;
  highRiskCount: number;
  mediumRiskCount: number;
  lowRiskCount: number;
  criticalCount: number;
  avgAttritionProb: number;
  fileName: string;
}

export type PipelineType = 'ibm' | 'synthetic';

export interface SchemaField {
  name: string;
  label: string;
  type: 'number' | 'categorical';
  options?: string[];
  min?: number;
  max?: number;
  default?: number | string;
  weight: number;
  description?: string;
}

export const IBM_SCHEMA: SchemaField[] = [
  { name: 'Age', label: 'Age', type: 'number', min: 18, max: 60, default: 35, weight: 0.3 },
  { name: 'BusinessTravel', label: 'Business Travel', type: 'categorical', options: ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], default: 'Travel_Rarely', weight: 0.6 },
  { name: 'DailyRate', label: 'Daily Rate', type: 'number', min: 100, max: 1500, default: 800, weight: 0.2 },
  { name: 'Department', label: 'Department', type: 'categorical', options: ['Research & Development', 'Sales', 'Human Resources'], default: 'Research & Development', weight: 0.4 },
  { name: 'DistanceFromHome', label: 'Distance From Home', type: 'number', min: 1, max: 29, default: 7, weight: 0.7 },
  { name: 'Education', label: 'Education (1-5)', type: 'number', min: 1, max: 5, default: 3, weight: 0.2 },
  { name: 'EducationField', label: 'Education Field', type: 'categorical', options: ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], default: 'Life Sciences', weight: 0.3 },
  { name: 'EnvironmentSatisfaction', label: 'Environment Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.8 },
  { name: 'Gender', label: 'Gender', type: 'categorical', options: ['Male', 'Female'], default: 'Male', weight: 0.1 },
  { name: 'HourlyRate', label: 'Hourly Rate', type: 'number', min: 30, max: 100, default: 65, weight: 0.15 },
  { name: 'JobInvolvement', label: 'Job Involvement (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.7 },
  { name: 'JobLevel', label: 'Job Level (1-5)', type: 'number', min: 1, max: 5, default: 2, weight: 0.4 },
  { name: 'JobRole', label: 'Job Role', type: 'categorical', options: ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], default: 'Research Scientist', weight: 0.5 },
  { name: 'JobSatisfaction', label: 'Job Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.9 },
  { name: 'MaritalStatus', label: 'Marital Status', type: 'categorical', options: ['Single', 'Married', 'Divorced'], default: 'Married', weight: 0.4 },
  { name: 'MonthlyIncome', label: 'Monthly Income', type: 'number', min: 1000, max: 20000, default: 6500, weight: 0.8 },
  { name: 'MonthlyRate', label: 'Monthly Rate', type: 'number', min: 2000, max: 27000, default: 14000, weight: 0.15 },
  { name: 'NumCompaniesWorked', label: 'Num Companies Worked', type: 'number', min: 0, max: 9, default: 2, weight: 0.5 },
  { name: 'OverTime', label: 'OverTime', type: 'categorical', options: ['Yes', 'No'], default: 'No', weight: 1.0 },
  { name: 'PercentSalaryHike', label: 'Percent Salary Hike', type: 'number', min: 11, max: 25, default: 15, weight: 0.3 },
  { name: 'PerformanceRating', label: 'Performance Rating (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.2 },
  { name: 'RelationshipSatisfaction', label: 'Relationship Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.5 },
  { name: 'StockOptionLevel', label: 'Stock Option Level (0-3)', type: 'number', min: 0, max: 3, default: 1, weight: 0.6 },
  { name: 'TotalWorkingYears', label: 'Total Working Years', type: 'number', min: 0, max: 40, default: 10, weight: 0.4 },
  { name: 'TrainingTimesLastYear', label: 'Training Times Last Year', type: 'number', min: 0, max: 6, default: 3, weight: 0.3 },
  { name: 'WorkLifeBalance', label: 'Work-Life Balance (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.8 },
  { name: 'YearsAtCompany', label: 'Years At Company', type: 'number', min: 0, max: 40, default: 5, weight: 0.5 },
  { name: 'YearsInCurrentRole', label: 'Years In Current Role', type: 'number', min: 0, max: 18, default: 3, weight: 0.4 },
  { name: 'YearsSinceLastPromotion', label: 'Years Since Last Promotion', type: 'number', min: 0, max: 15, default: 2, weight: 0.6 },
  { name: 'YearsWithCurrManager', label: 'Years With Current Manager', type: 'number', min: 0, max: 17, default: 3, weight: 0.4 },
];

export const SYNTHETIC_SCHEMA: SchemaField[] = [
  { name: 'age', label: 'Age', type: 'number', min: 18, max: 65, default: 35, weight: 0.3 },
  { name: 'distance_from_home', label: 'Distance From Home (miles)', type: 'number', min: 1, max: 30, default: 7, weight: 0.6 },
  { name: 'education', label: 'Education Level (1-5)', type: 'number', min: 1, max: 5, default: 3, weight: 0.2 },
  { name: 'environment_satisfaction', label: 'Environment Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.8 },
  { name: 'job_involvement', label: 'Job Involvement (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.7 },
  { name: 'job_level', label: 'Job Level (1-5)', type: 'number', min: 1, max: 5, default: 2, weight: 0.4 },
  { name: 'job_satisfaction', label: 'Job Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.9 },
  { name: 'monthly_income', label: 'Monthly Income ($)', type: 'number', min: 1000, max: 20000, default: 6500, weight: 0.8 },
  { name: 'monthly_rate', label: 'Monthly Rate ($)', type: 'number', min: 2000, max: 27000, default: 14000, weight: 0.15 },
  { name: 'num_companies_worked', label: 'Num Companies Worked', type: 'number', min: 0, max: 9, default: 2, weight: 0.5 },
  { name: 'percent_salary_hike', label: 'Percent Salary Hike (%)', type: 'number', min: 10, max: 25, default: 15, weight: 0.3 },
  { name: 'performance_rating', label: 'Performance Rating (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.2 },
  { name: 'relationship_satisfaction', label: 'Relationship Satisfaction (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.5 },
  { name: 'stock_option_level', label: 'Stock Option Level (0-3)', type: 'number', min: 0, max: 3, default: 1, weight: 0.6 },
  { name: 'total_working_years', label: 'Total Working Years', type: 'number', min: 0, max: 40, default: 10, weight: 0.4 },
  { name: 'training_times_last_year', label: 'Training Times Last Year', type: 'number', min: 0, max: 6, default: 3, weight: 0.3 },
  { name: 'work_life_balance', label: 'Work-Life Balance (1-4)', type: 'number', min: 1, max: 4, default: 3, weight: 0.8 },
  { name: 'years_at_company', label: 'Years At Company', type: 'number', min: 0, max: 40, default: 5, weight: 0.5 },
  { name: 'years_in_current_role', label: 'Years In Current Role', type: 'number', min: 0, max: 18, default: 3, weight: 0.4 },
  { name: 'years_since_last_promotion', label: 'Years Since Last Promotion', type: 'number', min: 0, max: 15, default: 2, weight: 0.6 },
];

export const IBM_COLUMN_NAMES = IBM_SCHEMA.map(f => f.name);
export const SYNTHETIC_COLUMN_NAMES = SYNTHETIC_SCHEMA.map(f => f.name);
