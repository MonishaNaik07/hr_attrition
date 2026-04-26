import { PredictionResult, PredictionRecord, BatchSummary, SchemaField, IBM_SCHEMA, SYNTHETIC_SCHEMA } from '../types';

// ---- Auth Service ----
const USERS_KEY = 'attrition_iq_users';
const SESSION_KEY = 'attrition_iq_session';

export function getUsers(): Record<string, string> {
  const raw = localStorage.getItem(USERS_KEY);
  return raw ? JSON.parse(raw) : {};
}

function saveUsers(users: Record<string, string>) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function registerUser(username: string, password: string): { success: boolean; message: string } {
  const users = getUsers();
  const normalizedKey = username.trim().toLowerCase();
  if (!normalizedKey) return { success: false, message: 'Username cannot be empty.' };
  if (users[normalizedKey]) return { success: false, message: 'Username already exists.' };
  users[normalizedKey] = password;
  saveUsers(users);
  return { success: true, message: 'Registration successful! You can now log in.' };
}

export function loginUser(username: string, password: string): { success: boolean; message: string } {
  const users = getUsers();
  const normalizedKey = username.trim().toLowerCase();
  if (!users[normalizedKey]) return { success: false, message: 'User not found. Please register first.' };
  if (users[normalizedKey] !== password) return { success: false, message: 'Incorrect password.' };
  localStorage.setItem(SESSION_KEY, JSON.stringify({ username: normalizedKey, loggedIn: true }));
  return { success: true, message: 'Login successful!' };
}

export function getCurrentUser(): { username: string; loggedIn: boolean } | null {
  const raw = localStorage.getItem(SESSION_KEY);
  return raw ? JSON.parse(raw) : null;
}

export function logoutUser() {
  localStorage.removeItem(SESSION_KEY);
}

// ---- History Storage ----
const HISTORY_KEY = 'attrition_iq_history';
const BATCH_KEY = 'attrition_iq_batches';

export function getHistory(): PredictionRecord[] {
  const raw = localStorage.getItem(HISTORY_KEY);
  return raw ? JSON.parse(raw) : [];
}

export function saveRecord(record: PredictionRecord) {
  const history = getHistory();
  history.unshift(record);
  if (history.length > 500) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function getBatches(): BatchSummary[] {
  const raw = localStorage.getItem(BATCH_KEY);
  return raw ? JSON.parse(raw) : [];
}

export function saveBatch(batch: BatchSummary) {
  const batches = getBatches();
  batches.unshift(batch);
  if (batches.length > 100) batches.pop();
  localStorage.setItem(BATCH_KEY, JSON.stringify(batches));
}

export function clearAllHistory() {
  localStorage.removeItem(HISTORY_KEY);
  localStorage.removeItem(BATCH_KEY);
}

// ---- Mock Prediction Engine ----
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function normalize(val: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return (val - min) / (max - min);
}

function generateRiskFactors(input: Record<string, string | number>, _schema: SchemaField[]): string[] {
  const factors: string[] = [];
  const get = (key: string) => Number(input[key] || 0);
  const getStr = (key: string) => String(input[key] || '');

  // Overtime check
  if (getStr('OverTime') === 'Yes' || getStr('overtime') === 'Yes') {
    factors.push('⚡ Employee is working overtime — strong burnout indicator');
  }

  // Job satisfaction
  const js = get('JobSatisfaction') || get('job_satisfaction');
  if (js <= 2) {
    factors.push('📉 Low job satisfaction score detected — disengagement risk');
  }

  // Environment satisfaction
  const es = get('EnvironmentSatisfaction') || get('environment_satisfaction');
  if (es <= 2) {
    factors.push('🏢 Poor work environment satisfaction — cultural misalignment');
  }

  // Work-life balance
  const wlb = get('WorkLifeBalance') || get('work_life_balance');
  if (wlb <= 2) {
    factors.push('⚖️ Poor work-life balance — high stress indicator');
  }

  // Monthly income
  const mi = get('MonthlyIncome') || get('monthly_income');
  if (mi < 3500) {
    factors.push('💰 Below-market compensation — competitive attrition risk');
  }

  // Distance from home
  const dfh = get('DistanceFromHome') || get('distance_from_home');
  if (dfh > 15) {
    factors.push('🚗 Long commute distance — commute fatigue detected');
  }

  // Years since last promotion
  const yslp = get('YearsSinceLastPromotion') || get('years_since_last_promotion');
  if (yslp > 5) {
    factors.push('📈 Stagnant career progression — no recent promotion');
  }

  // Job involvement
  const ji = get('JobInvolvement') || get('job_involvement');
  if (ji <= 2) {
    factors.push('🎯 Low job involvement — role disconnection');
  }

  // Num companies worked
  const ncw = get('NumCompaniesWorked') || get('num_companies_worked');
  if (ncw >= 5) {
    factors.push('🔄 High job-hopping history — pattern of frequent transitions');
  }

  // Business travel
  const bt = getStr('BusinessTravel') || getStr('business_travel');
  if (bt === 'Travel_Frequently') {
    factors.push('✈️ Frequent business travel — elevated stress & burnout');
  }

  // Stock options
  const so = get('StockOptionLevel') || get('stock_option_level');
  if (so === 0) {
    factors.push('📊 No stock options — weak financial retention anchor');
  }

  // Total working years
  const twy = get('TotalWorkingYears') || get('total_working_years');
  if (twy < 2) {
    factors.push('🌱 Early-career employee — higher mobility tendency');
  }

  // Years at company
  const yac = get('YearsAtCompany') || get('years_at_company');
  if (yac < 1) {
    factors.push('🕐 New hire in first year — onboarding attrition window');
  }

  // Relationship satisfaction
  const rs = get('RelationshipSatisfaction') || get('relationship_satisfaction');
  if (rs <= 2) {
    factors.push('🤝 Poor workplace relationships — social disconnection');
  }

  if (factors.length === 0) {
    factors.push('✅ No major risk factors detected — employee appears stable');
  }

  return factors;
}

function generateRetentionActions(input: Record<string, string | number>, riskLevel: string): string[] {
  const actions: string[] = [];
  const get = (key: string) => Number(input[key] || 0);
  const getStr = (key: string) => String(input[key] || '');

  if (riskLevel === 'Low') {
    actions.push('🌟 Continue current engagement strategy — maintain momentum');
    actions.push('📋 Schedule quarterly check-ins to sustain satisfaction levels');
    return actions;
  }

  // Overtime
  if (getStr('OverTime') === 'Yes' || getStr('overtime') === 'Yes') {
    actions.push('🕐 Implement workload redistribution — reduce overtime immediately');
    actions.push('💰 Introduce overtime compensation or time-off-in-lieu policy');
  }

  // Low satisfaction
  const js = get('JobSatisfaction') || get('job_satisfaction');
  if (js <= 2) {
    actions.push('🎯 Launch targeted job enrichment program with new responsibilities');
    actions.push('🗣️ Schedule 1-on-1 career development discussion within 2 weeks');
  }

  // Income
  const mi = get('MonthlyIncome') || get('monthly_income');
  if (mi < 3500) {
    actions.push('💵 Conduct immediate compensation benchmarking analysis');
    actions.push('📊 Propose salary adjustment or performance-based bonus structure');
  }

  // Work-life balance
  const wlb = get('WorkLifeBalance') || get('work_life_balance');
  if (wlb <= 2) {
    actions.push('🏠 Offer hybrid/remote work arrangements');
    actions.push('📅 Implement flexible scheduling or compressed work weeks');
  }

  // Career stagnation
  const yslp = get('YearsSinceLastPromotion') || get('years_since_last_promotion');
  if (yslp > 5) {
    actions.push('🚀 Create accelerated career development roadmap');
    actions.push('🎓 Fund relevant certification or leadership training');
  }

  // Commute
  const dfh = get('DistanceFromHome') || get('distance_from_home');
  if (dfh > 15) {
    actions.push('🏠 Evaluate remote work eligibility (2-3 days/week)');
    actions.push('🚌 Offer transportation subsidy or relocation assistance');
  }

  // Travel
  const bt = getStr('BusinessTravel') || getStr('business_travel');
  if (bt === 'Travel_Frequently') {
    actions.push('✈️ Reduce travel frequency — shift to virtual meetings');
    actions.push('🏨 Upgrade travel accommodations and travel perks');
  }

  // Stock options
  const so = get('StockOptionLevel') || get('stock_option_level');
  if (so === 0) {
    actions.push('📈 Introduce equity/stock option grant to increase retention');
  }

  // Environment
  const es = get('EnvironmentSatisfaction') || get('environment_satisfaction');
  if (es <= 2) {
    actions.push('🏢 Conduct workplace environment assessment');
    actions.push('☕ Upgrade amenities — workspace, break rooms, collaboration areas');
  }

  if (actions.length === 0) {
    actions.push('📋 Schedule retention review meeting with HR within 30 days');
    actions.push('🎯 Assign a mentor or buddy for increased engagement');
  }

  if (riskLevel === 'Critical') {
    actions.unshift('🚨 URGENT: Schedule immediate retention intervention meeting');
    actions.unshift('📞 Escalate to HR Director for executive retention package');
  }

  return [...new Set(actions)].slice(0, 8);
}

export function predictAttrition(
  input: Record<string, string | number>,
  pipeline: 'ibm' | 'synthetic'
): PredictionResult {
  const schema = pipeline === 'ibm' ? IBM_SCHEMA : SYNTHETIC_SCHEMA;

  // Compute weighted risk score
  let rawScore = 0;
  let totalWeight = 0;

  for (const field of schema) {
    const val = input[field.name];
    if (val === undefined || val === null) continue;

    totalWeight += field.weight;

    if (field.type === 'number') {
      const numVal = Number(val);
      const normalized = normalize(numVal, field.min || 0, field.max || 100);

      // Invert satisfaction-like fields (low = risky)
      const invertFields = ['JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction',
        'WorkLifeBalance', 'RelationshipSatisfaction', 'PerformanceRating',
        'job_satisfaction', 'job_involvement', 'environment_satisfaction',
        'work_life_balance', 'relationship_satisfaction', 'performance_rating',
        'percent_salary_hike', 'PercentSalaryHike', 'stock_option_level', 'StockOptionLevel'];

      const directFields = ['DistanceFromHome', 'NumCompaniesWorked', 'YearsSinceLastPromotion',
        'distance_from_home', 'num_companies_worked', 'years_since_last_promotion'];

      let contribution: number;
      if (invertFields.includes(field.name)) {
        contribution = (1 - normalized) * field.weight;
      } else if (directFields.includes(field.name)) {
        contribution = normalized * field.weight;
      } else {
        // For income-like fields, low values are risky
        const incomeFields = ['MonthlyIncome', 'monthly_income', 'DailyRate', 'DailyRate',
          'HourlyRate', 'hourly_rate', 'MonthlyRate', 'monthly_rate'];
        if (incomeFields.includes(field.name)) {
          contribution = (1 - normalized) * field.weight * 0.7;
        } else {
          contribution = (1 - normalized) * field.weight * 0.3;
        }
      }
      rawScore += contribution;
    } else if (field.type === 'categorical') {
      const strVal = String(val);
      // Categorical risk mapping
      const catRisk: Record<string, number> = {
        'Travel_Frequently': 0.9,
        'Travel_Rarely': 0.3,
        'Non-Travel': 0.1,
        'Sales': 0.6,
        'Research & Development': 0.3,
        'Sales Representative': 0.85,
        'Laboratory Technician': 0.7,
        'Human Resources': 0.6,
        'Research Scientist': 0.4,
        'Sales Executive': 0.5,
        'Healthcare Representative': 0.35,
        'Manufacturing Director': 0.2,
        'Manager': 0.15,
        'Research Director': 0.1,
        'Single': 0.7,
        'Married': 0.3,
        'Divorced': 0.5,
        'Male': 0.45,
        'Female': 0.45,
        'Yes': 0.9,
        'No': 0.2,
        'Marketing': 0.5,
        'Technical Degree': 0.5,
        'Life Sciences': 0.35,
        'Medical': 0.35,
        'Other': 0.4,
      };
      const risk = catRisk[strVal] ?? 0.4;
      rawScore += risk * field.weight;
    }
  }

  // Normalize and apply sigmoid
  const avgScore = totalWeight > 0 ? rawScore / totalWeight : 0.3;
  const sigmoidInput = (avgScore - 0.35) * 8; // Center around 0.35 threshold
  let probability = sigmoid(sigmoidInput);

  // Add small random variation for realism (±3%)
  probability = Math.max(0.02, Math.min(0.98, probability + (Math.random() - 0.5) * 0.06));

  // Determine risk level
  let riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
  if (probability < 0.25) riskLevel = 'Low';
  else if (probability < 0.5) riskLevel = 'Medium';
  else if (probability < 0.75) riskLevel = 'High';
  else riskLevel = 'Critical';

  const riskFactors = generateRiskFactors(input, schema);
  const retentionActions = generateRetentionActions(input, riskLevel);

  return {
    attritionProbability: Math.round(probability * 1000) / 10,
    riskLevel,
    riskFactors,
    retentionActions,
    pipeline,
  };
}

// ---- Batch Processing ----
export function detectPipeline(columns: string[]): 'ibm' | 'synthetic' | null {
  const cols = columns.map(c => c.trim());
  const ibmMatches = cols.filter(c =>
    IBM_SCHEMA.some(f => f.name.toLowerCase() === c.toLowerCase())
  ).length;
  const synthMatches = cols.filter(c =>
    SYNTHETIC_SCHEMA.some(f => f.name.toLowerCase() === c.toLowerCase())
  ).length;

  if (ibmMatches > synthMatches && ibmMatches >= 10) return 'ibm';
  if (synthMatches > ibmMatches && synthMatches >= 10) return 'synthetic';
  if (ibmMatches >= synthMatches) return 'ibm';
  return 'synthetic';
}

export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

export function getStats() {
  const history = getHistory();
  const batches = getBatches();
  const totalPredictions = history.length;
  const highRisk = history.filter(h => h.result.riskLevel === 'High' || h.result.riskLevel === 'Critical').length;
  const avgProb = totalPredictions > 0
    ? Math.round(history.reduce((s, h) => s + h.result.attritionProbability, 0) / totalPredictions * 10) / 10
    : 0;
  return {
    totalPredictions,
    highRisk,
    avgProb,
    totalBatches: batches.length,
    totalRecords: batches.reduce((s, b) => s + b.totalRecords, 0),
  };
}
