document.addEventListener('DOMContentLoaded', () => {
    
    // --- NAVIGATION LOGIC ---
    loadRecords();

    // --- NAVIGATION LOGIC ---

    // --- NAVIGATION LOGIC ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.content-view');

    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = btn.getAttribute('data-target');
            
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            views.forEach(v => v.classList.remove('active-view'));
            document.getElementById(targetId).classList.add('active-view');

            if (targetId === 'records') loadRecords();
        });
    });

    // --- PIPELINE TOGGLE LOGIC ---
    const modelTypeSelector = document.getElementById('model_type');
    const ibmFields = document.getElementById('ibm_fields');
    const synFields = document.getElementById('syn_fields');

    if (modelTypeSelector) {
        // Init state
        Array.from(synFields.querySelectorAll('input, select')).forEach(i => i.disabled = true);

        modelTypeSelector.addEventListener('change', (e) => {
            if (e.target.value === 'ibm') {
                ibmFields.style.display = 'grid';
                synFields.style.display = 'none';
                Array.from(ibmFields.querySelectorAll('input, select')).forEach(i => i.disabled = false);
                Array.from(synFields.querySelectorAll('input, select')).forEach(i => i.disabled = true);
            } else {
                ibmFields.style.display = 'none';
                synFields.style.display = 'grid';
                Array.from(ibmFields.querySelectorAll('input, select')).forEach(i => i.disabled = true);
                Array.from(synFields.querySelectorAll('input, select')).forEach(i => i.disabled = false);
            }
        });
    }

    // --- INDIVIDUAL PREDICTION ---
    const predForm = document.getElementById('prediction-form');
    const explainZone = document.getElementById('explain-zone');

    predForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btnText = document.getElementById('btn-text');
        const btnLoader = document.getElementById('btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
        
        const data = Object.fromEntries(new FormData(predForm).entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            // Update local dashboard counter directly for speed
            const dashCount = document.getElementById('dash-analyzed');
            dashCount.innerText = parseInt(dashCount.innerText) + 1;

            // Silently update records in background
            loadRecords(true); 

            const statusBox = document.getElementById('prediction-status');
            const statusText = document.getElementById('prediction-text');
            statusBox.className = 'prediction-status';
            
            if (result.raw_prediction === 1) {
                statusBox.classList.add('status-danger');
                statusText.innerText = `Flight Risk (${result.attrition_probability}%)`;
                
                if (result.explanation) {
                explainZone.style.display = 'block';
                const resList = document.getElementById('explain-reasons');
                const solList = document.getElementById('explain-solutions');
                
                resList.innerHTML = '';
                solList.innerHTML = '';
                
                // New logic: combined tailored recommendations
                result.explanation.forEach(rec => {
                    const li = document.createElement('li');
                    li.style.borderLeft = '4px solid var(--blue-primary)';
                    li.style.paddingLeft = '1rem';
                    li.style.marginBottom = '1rem';
                    li.innerHTML = `
                        <div style="font-weight:600; font-size:0.9rem; color:var(--navy);">Why: ${rec.reason}</div>
                        <div style="font-size:0.95rem; color:var(--blue-primary); margin-top:0.25rem;">Action: ${rec.solution}</div>
                    `;
                    resList.appendChild(li);
                });
                
                // Hide the separate solutions list header as it's now integrated
                document.querySelector('h4[style*="color: var(--success)"]').style.display = 'none';
            }
            } else {
                statusBox.classList.add('status-success');
                statusText.innerText = `Stable Retention (${result.stay_probability}%)`;
                explainZone.style.display = 'none';
            }
        } catch (err) {
            alert('Error running profile.');
        } finally {
            btnText.style.display = 'inline-block';
            btnLoader.style.display = 'none';
        }
    });

    // --- CLEAR FORM LOGIC ---
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            predForm.reset();
            
            // Hide results
            const resultCard = document.getElementById('result-card');
            const statusBox = document.getElementById('prediction-status');
            const statusText = document.getElementById('prediction-text');
            const explainZone = document.getElementById('explain-zone');
            
            statusBox.className = 'prediction-status';
            statusText.innerText = 'Waiting...';
            explainZone.style.display = 'none';
            
            // Refill with random data for convenience if desired (optional)
            // refillMockData();
        });
    }

    // --- BATCH UPLOAD LOGIC ---
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('csv-file');
    const batchResults = document.getElementById('batch-results');
    const batchTbody = document.getElementById('batch-tbody');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if(!fileInput.files[0]) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        uploadForm.querySelector('button').innerText = 'Processing...';
        
        try {
            const res = await fetch('/upload_csv', { method: 'POST', body: formData });
            const data = await res.json();
            
            if(data.success) {
                // Update dashboard badge
                const dashCount = document.getElementById('dash-analyzed');
                dashCount.innerText = parseInt(dashCount.innerText) + data.total;

                batchTbody.innerHTML = '';
                data.results.forEach(r => {
                    const tr = document.createElement('tr');
                    let factor = r.details ? r.details.reasons[0] : 'Stable Factors';
                    if(!r.details) factor = '<span style="color:var(--success)">Stable Factors</span>';
                    else factor = `<span style="color:var(--danger)">${factor}</span>`;

                    tr.innerHTML = `
                        <td>#${r.row_index}</td>
                        <td><span style="color: ${r.prediction==='Attrited'?'var(--danger)':'var(--success)'}; font-weight:600;">${r.prediction}</span></td>
                        <td>${r.risk}%</td>
                        <td>${factor}</td>
                    `;
                    batchTbody.appendChild(tr);
                });
                batchResults.style.display = 'block';
            } else {
                alert('Dataset Error: ' + data.error);
            }
        } catch(err) {
            alert('Error parsing CSV');
        } finally {
            uploadForm.querySelector('button').innerText = 'Process Batch';
        }
    });

    // --- RECORDS LOGIC ---
    let allRecords = [];
    let displayedCount = 0;
    const PAGE_SIZE = 500;
    const modal = document.getElementById('record-detail-modal');
    const modalBody = document.getElementById('modal-body');
    const closeModal = document.querySelector('.close-modal');

    async function loadRecords(isSilent = false) {
        const tb = document.getElementById('records-tbody');
        const loadMoreBtn = document.getElementById('load-more-btn');
        const totalCountBadge = document.getElementById('total-records-count');
        const dashCount = document.getElementById('dash-analyzed');
        
        try {
            const res = await fetch('/records');
            allRecords = await res.json();
            
            tb.innerHTML = '';
            displayedCount = 0;
            totalCountBadge.innerText = allRecords.length;
            if (dashCount) dashCount.innerText = allRecords.length;

            if (allRecords.length === 0) {
                tb.innerHTML = '<tr><td colspan="3" style="text-align:center">No records history yet.</td></tr>';
                loadMoreBtn.style.display = 'none';
                return;
            }

            renderNextBatch();
            
        } catch(e) {
            console.error("Error loading records:", e);
        }
    }

    function renderNextBatch() {
        const tb = document.getElementById('records-tbody');
        const loadMoreBtn = document.getElementById('load-more-btn');
        
        const nextBatch = allRecords.slice(displayedCount, displayedCount + PAGE_SIZE);
        
        nextBatch.forEach((r, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>Analysis #${r.id || (allRecords.length - displayedCount - idx)}</strong></td>
                <td><span style="color: ${r.prediction==='Attrited'?'var(--danger)':'var(--success)'}; font-weight:600;">${r.prediction}</span></td>
                <td style="font-size: 0.8rem; color: var(--text-muted);">${r.timestamp || 'N/A'}</td>
            `;
            
            // Add click listener for details
            tr.addEventListener('click', () => showRecordDetails(r));
            tb.appendChild(tr);
        });

        displayedCount += nextBatch.length;
        
        loadMoreBtn.style.display = (displayedCount < allRecords.length) ? 'flex' : 'none';
    }

    // Back to Top Logic
    const backToTopBtn = document.getElementById('back-to-top');
    const mainContent = document.querySelector('.main-content');

    if (mainContent) {
        mainContent.addEventListener('scroll', () => {
            if (mainContent.scrollTop > 300) {
                backToTopBtn.style.display = 'flex';
            } else {
                backToTopBtn.style.display = 'none';
            }
        });
    }

    backToTopBtn.addEventListener('click', () => {
        mainContent.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function showRecordDetails(record) {
        modalBody.innerHTML = '';
        
        // Header Section
        const header = document.createElement('div');
        header.style.gridColumn = '1 / -1';
        header.style.padding = '1rem';
        header.style.background = 'var(--bg-color)';
        header.style.borderRadius = '8px';
        header.style.marginBottom = '1rem';
        header.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="color:var(--blue-primary);">${record.employee_name || 'Anonymous Profile'}</h2>
                <span class="stat-badge" style="background:${record.prediction==='Attrited'?'var(--danger)':'var(--success)'}; color:white; padding:0.4rem 1rem; border-radius:20px;">${record.prediction}</span>
            </div>
            <p style="margin-top:0.5rem; color:var(--text-muted);">Analyzed via ${record.model_type || 'Standard'} Pipeline on ${record.timestamp}</p>
        `;
        modalBody.appendChild(header);

        // Confidence Logic
        const confBox = document.createElement('div');
        confBox.style.gridColumn = '1 / -1';
        confBox.className = 'detail-item';
        confBox.style.border = '2px solid var(--blue-primary)';
        confBox.style.background = '#eff6ff';
        
        const confText = record.prediction === 'Attrited' ? 
            `The model is ${record.confidence}% certain this employee is a flight risk. This high confidence is driven by patterns in ${record.explanation[0].reason || 'multiple performance metrics'}.` :
            `The model is ${record.confidence}% certain this employee will stay. The high stability score is linked to favorable indicators in compensation and job involvement.`;

        confBox.innerHTML = `
            <div class="detail-label">AI Confidence Explanation</div>
            <div class="detail-value" style="font-size:1.1rem; line-height:1.4;">${confText}</div>
        `;
        modalBody.appendChild(confBox);

        // All Features
        const exclude = [
            'role', 'prediction', 'timestamp', 'employee_name', 'model_type', 'explanation', 
            'Age', 'MonthlyIncome', 'Department', 'Attrition', 'DistanceFromHome', 
            'Over18', 'StandardHours', 'EmployeeCount', 'EmployeeNumber',
            'DailyRate', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate', 
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'confidence'
        ];
        Object.keys(record).forEach(key => {
            if (exclude.includes(key) || record[key] === undefined) return;
            
            const item = document.createElement('div');
            item.className = 'detail-item';
            
            let label = key.replace(/([A-Z])/g, ' $1').trim();
            let value = record[key];
            
            if (key === 'confidence') label = 'Exact Probability (%)';
            if (key === 'MonthlyIncome') value = `$${value}`;
            
            item.innerHTML = `
                <div class="detail-label">${label}</div>
                <div class="detail-value">${value}</div>
            `;
            modalBody.appendChild(item);
        });

        // Detailed Tailored Recommendations
        if(record.explanation) {
            const expBox = document.createElement('div');
            expBox.style.gridColumn = '1 / -1';
            expBox.style.marginTop = '1.5rem';
            expBox.style.padding = '1.5rem';
            expBox.style.background = '#f1f5f9';
            expBox.style.borderRadius = '12px';
            
            expBox.innerHTML = `
                <div class="detail-label" style="margin-bottom:1rem; font-size:1rem; color:var(--navy);">Individual Retention Strategy</div>
                <div id="modal-recs-list"></div>
            `;
            modalBody.appendChild(expBox);
            
            const recList = expBox.querySelector('#modal-recs-list');
            record.explanation.forEach(rec => {
                const item = document.createElement('div');
                item.style.marginBottom = '1.25rem';
                item.style.padding = '1rem';
                item.style.background = 'white';
                item.style.borderRadius = '8px';
                item.style.borderLeft = '5px solid var(--blue-primary)';
                item.innerHTML = `
                    <div style="font-size:0.85rem; color:var(--text-muted); font-weight:600;">DETECTED TRIGGER</div>
                    <div style="font-size:1rem; color:var(--navy); margin-bottom:0.5rem;">${rec.reason}</div>
                    <div style="font-size:0.85rem; color:var(--blue-primary); font-weight:600;">TAILORED SOLUTION</div>
                    <div style="font-size:1rem; color:var(--blue-primary);">${rec.solution}</div>
                `;
                recList.appendChild(item);
            });
        }

        modal.style.display = 'flex';
    }

    // Modal closing logic
    if (closeModal) {
        closeModal.onclick = () => modal.style.display = 'none';
    }
    window.onclick = (event) => {
        if (event.target == modal) modal.style.display = 'none';
    }

    document.getElementById('load-more-btn').addEventListener('click', renderNextBatch);

    // --- CLEAR HISTORY LOGIC ---
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            if (confirm("Are you sure you want to permanently delete all prediction history? This action cannot be undone.")) {
                try {
                    const res = await fetch('/clear_records', { method: 'POST' });
                    const data = await res.json();
                    if (data.success) {
                        loadRecords();
                        // Reset dashboard counter too
                        const dashCount = document.getElementById('dash-analyzed');
                        if (dashCount) dashCount.innerText = "0";
                    }
                } catch (e) {
                    alert("Error clearing history.");
                }
            }
        });
    }

    // Mock initial data filling for speed
    const inputs = document.querySelectorAll("input[type=number]");
    inputs.forEach(input => {
        if (!input.value) {
            if (input.max === "5") input.value = Math.floor(Math.random() * 5) + 1;
            else if (input.id.includes("Age")) input.value = Math.floor(Math.random() * 30) + 25;
            else if (input.id.includes("Income") || input.id.includes("Rate")) input.value = Math.floor(Math.random() * 5000) + 4000;
            else if (input.id.includes("Years") || input.id.includes("Distance")) input.value = Math.floor(Math.random() * 15) + 1;
            else input.value = Math.floor(Math.random() * 10) + 1;
        }
    });

});
