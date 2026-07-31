/**
 * Novel Agent V7 — 创作流水线
 */
const App = {
    currentProjectId: null,
    ws: null,
    _assistantContext: null,
    _projects: [],
    _workflowDone: false,
    _pipelineState: {},  // step -> 'pending' | 'active' | 'running' | 'done'

    async init() {
        this._setupNav();
        this._setupFormHandlers();
        this._setupGlobalClicks();
        this._updateApiStatus();
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && document.activeElement?.id === 'chat-input') {
                e.preventDefault(); this.sendAssistantMessage();
            }
        });
        await this._refreshProjects();
        this._navigate('workspace');
    },

    _setupGlobalClicks() {
        // Event delegation for project cards
        const cardsEl = document.getElementById('my-works-cards');
        if (cardsEl) {
            cardsEl.addEventListener('click', (e) => {
                const card = e.target.closest('[data-project-id]');
                if (card) this.openProject(card.dataset.projectId);
            });
        }
    },

    // ================================================================
    // Navigation
    // ================================================================
    _setupNav() {
        window.addEventListener('hashchange', () => this._route());
        ['nav-workspace','nav-my-works','nav-assistant'].forEach(id => {
            document.getElementById(id).addEventListener('click', (e) => {
                e.preventDefault();
                const v = id.replace('nav-','');
                if (v === 'my-works') { this.currentProjectId = null; if (this.ws) { this.ws.close(); this.ws = null; } }
                this._navigate(v);
            });
        });
    },

    _navigate(viewName) {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.querySelectorAll('#nav .nav-item').forEach(n => n.classList.remove('active'));
        const el = document.getElementById(`view-${viewName}`);
        if (el) el.classList.add('active');
        const navId = viewName === 'project' ? 'nav-my-works' : `nav-${viewName}`;
        const nav = document.getElementById(navId);
        if (nav) nav.classList.add('active');
        // Use replaceState to avoid triggering hashchange again
        if (window.location.hash !== `#${viewName}`) {
            history.replaceState(null, '', `#${viewName}`);
        }
        if (viewName === 'my-works') this._renderProjectList();
        if (viewName === 'project' && this.currentProjectId) this._loadProjectDetail();
        // Show/hide workspace header based on active project
        if (viewName === 'workspace') {
            if (this.currentProjectId) {
                // Project is loaded — keep header as is (set by openProject or startScan)
            } else {
                // No project — show default title, hide project header
                document.getElementById('workspace-default-title').style.display = '';
                document.getElementById('workspace-project-header').style.display = 'none';
            }
        }
    },

    _route() {
        const v = (window.location.hash || '#workspace').replace('#','');
        if (v === 'project') {
            if (this.currentProjectId) { this._navigate('project'); }
            else { this._navigate('my-works'); }
        } else {
            this._navigate(v);
        }
    },

    // ================================================================
    // 我的书库 — 纯项目列表
    // ================================================================
    async _refreshProjects() {
        try { this._projects = await API.listProjects() || []; }
        catch (err) { console.error(err); }
    },

    _renderProjectList() {
        const container = document.getElementById('my-works-cards');
        if (!container) return;
        if (!this._projects.length) {
            container.innerHTML = '<p class="empty-state">暂无项目，前往<a href="#workspace">工作台</a>开始创作</p>';
            return;
        }
        container.innerHTML = this._projects.map(p => {
            const badge = p.current_chapter && p.total_chapters ? `${p.current_chapter}/${p.total_chapters}章` : '';
            return `<div class="card" data-project-id="${p.project_id}" style="cursor:pointer">
                <h3>${this._esc(p.title || '未命名')}</h3>
                <div class="meta">${badge || '尚未开始'}</div>
                <span class="status status-${p.status||'initialized'}">${p.status||'initialized'}</span>
            </div>`;
        }).join('');
    },

    async openProject(id) {
        this.currentProjectId = id;
        this._workflowDone = false;
        this._connectWS(id);
        this._initPipeline();
        try {
            const p = await API.getProject(id);
            const title = p.config?.title || '未命名';
            this._showProjectHeader(title);
            const meta = p.meta;
            if (meta && meta.current_phase && meta.current_phase !== 'idle') {
                this._restorePipelineState(meta);
            }
        } catch(e) { /* ignore */ }
        this._navigate('workspace');
    },

    async _loadProjectDetail() {
        if (!this.currentProjectId) return;
        try {
            const p = await API.getProject(this.currentProjectId);
            document.getElementById('project-title').textContent = p.config.title || '未命名';
            document.getElementById('overview-content').innerHTML = Components.overviewEditable(p);
            // Detect if already complete
            const tc = p.meta?.total_chapters || 0;
            const cc = p.meta?.current_chapter || 0;
            if (tc > 0 && cc >= tc) this._workflowDone = true;
            this._setupTabs();
        } catch (err) { console.error(err); }
    },

    // ================================================================
    // Tabs
    // ================================================================
    _setupTabs() {
        document.querySelectorAll('#view-project .tab').forEach(tab => {
            tab.onclick = async () => {
                document.querySelectorAll('#view-project .tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('#view-project .tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(`tab-${tab.dataset.tab}`)?.classList.add('active');
                await this._loadTabContent(tab.dataset.tab);
            };
        });
    },

    async _loadTabContent(tab) {
        if (!this.currentProjectId) return;
        try {
            if (tab === 'topics') { document.getElementById('topics-content').innerHTML = Components.topicResearchView(); }
            else if (tab === 'bible') { const d = await API.getBible(this.currentProjectId); document.getElementById('bible-content').innerHTML = Components.bibleEditable(d); }
            else if (tab === 'characters') { const d = await API.getCharacters(this.currentProjectId); document.getElementById('characters-content').innerHTML = Components.charactersEditable(d); }
            else if (tab === 'outline') { const d = await API.getOutline(this.currentProjectId); document.getElementById('outline-content').innerHTML = Components.outlineEditable(d); }
            else if (tab === 'chapters') { const d = await API.getChapters(this.currentProjectId); let tc = 0; try { const p = await API.getProject(this.currentProjectId); tc = p.meta?.total_chapters || 0; } catch(e){} document.getElementById('chapters-content').innerHTML = Components.chaptersView(d, {workflowDone: this._workflowDone, totalChapters: tc}); }
        } catch (err) { if (!err.message?.includes('404')) console.error(err); }
    },

    async viewChapter(num) {
        try {
            const ch = await API.getChapter(this.currentProjectId, num);
            const cn = Components._CN_NUM[num] || num;
            let html = `<h3>第${cn}章 ${Components._esc(ch.plan?.title||ch.title||'')}</h3>`;
            // Show status
            const statusLabels = { planned:'已规划', writing:'写作中', reviewing:'审核中', polishing:'润色中', done:'已完成' };
            if (ch.status) html += `<p>状态：${statusLabels[ch.status]||ch.status}</p>`;
            // Show review if available
            if (ch.review) html += Components.reviewScores(ch.review.dimension_scores);
            if (ch.review?.issues) html += Components.issuesList(ch.review.issues);
            html += '<hr style="margin:16px 0;border-color:var(--border)">';
            // Show content or explain what's missing
            if (ch.content) {
                let cleanContent = ch.content.replace(/^#{1,4}\s+.+$/gm, '');
                cleanContent = cleanContent.replace(/\n{3,}/g, '\n\n').trim();
                html += `<div style="display:flex;align-items:center;gap:8px;margin-top:12px"><strong>正文</strong><button class="btn btn-secondary btn-sm" onclick="App.toggleChapterEdit(${num})">编辑</button></div>`;
                html += `<div id="chapter-display-${num}" style="white-space:pre-wrap;line-height:2;font-size:1rem;margin-top:8px">${Components._esc(cleanContent)}</div>`;
                html += `<div id="chapter-editor-${num}" style="display:none"><textarea id="chapter-textarea-${num}" style="width:100%;min-height:300px;padding:12px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);font-family:inherit;font-size:0.95rem;line-height:1.8;resize:vertical">${Components._esc(cleanContent)}</textarea><button class="btn btn-success btn-sm" onclick="App.saveChapterEdit(${num})" style="margin-top:8px">保存</button></div>`;
            } else if (ch.plan) {
                html += `<p style="color:var(--text-secondary)">章节已规划，内容尚未生成。</p>`;
                if (ch.plan.goal) html += `<p><strong>本章目标：</strong>${Components._esc(ch.plan.goal)}</p>`;
                if (ch.plan.scenes?.length) {
                    const sceneTexts = ch.plan.scenes.map(s => {
                        if (typeof s === 'string') {
                            // Clean raw markers: 【xxx】text（POV → just text
                            let t = s.replace(/【[^】]+】/g, '').replace(/（POV.*?）|\(POV.*?\)/g, '').replace(/→/g, '').trim();
                            return Components._esc(t.substring(0, 80));
                        }
                        return Components._esc((s.setting || s.goal || '').substring(0, 80));
                    }).filter(Boolean);
                    if (sceneTexts.length) html += `<p><strong>场景：</strong>${sceneTexts.join(' → ')}</p>`;
                }
                html += `<button class="btn btn-primary btn-sm" onclick="App.startWorkflow()" style="margin-top:8px">继续生成</button>`;
            } else {
                html += '<p style="color:var(--text-secondary)">该章节尚未生成。点击「继续生成」开始写作。</p>';
            }
            if (ch.content) {
                html += `<div style="margin-top:16px"><a href="${API.getChapterMdUrl(this.currentProjectId, num)}" class="btn btn-secondary btn-sm">下载 Markdown</a></div>`;
            }
            document.getElementById('chapter-modal-body').innerHTML = html;
            document.getElementById('chapter-modal').classList.add('active');
            document.querySelector('.modal-close').onclick = () => document.getElementById('chapter-modal').classList.remove('active');
            document.getElementById('chapter-modal').onclick = (e) => { if (e.target === document.getElementById('chapter-modal')) document.getElementById('chapter-modal').classList.remove('active'); };
        } catch (err) { alert(`加载失败: ${err.message}`); }
    },

    exportDocx() { if (this.currentProjectId) window.open(API.getExportDocxUrl(this.currentProjectId)); },
    exportMarkdown() { if (this.currentProjectId) window.open(API.getExportMdUrl(this.currentProjectId)); },

    // ================================================================
    // Inline Edit — generic toggle for any section
    // ================================================================
    toggleEdit(sectionId) {
        const display = document.getElementById(`${sectionId}-display`);
        const editor = document.getElementById(`${sectionId}-editor`);
        const btn = document.getElementById(`${sectionId}-edit-btn`);
        if (!display || !editor) return;
        const editing = editor.style.display !== 'none';
        if (editing) {
            const textarea = editor.querySelector('textarea');
            if (textarea) {
                const newValue = textarea.value;
                // Keep same font: update display with pre-wrap to match textarea
                display.innerHTML = `<div style="white-space:pre-wrap;font-family:inherit;font-size:0.95rem;line-height:1.7">${Components._esc(newValue)}</div>`;
                // Save to backend
                this._saveEdit(sectionId, newValue);
            }
            editor.style.display = 'none'; display.style.display = '';
            btn.textContent = '编辑';
        } else {
            const textarea = editor.querySelector('textarea');
            if (textarea) textarea.value = display.innerText;
            editor.style.display = ''; display.style.display = 'none';
            btn.textContent = '保存';
        }
    },

    async _saveEdit(sectionId, value) {
        if (!this.currentProjectId) return;
        try {
            // Determine artifact type from section ID prefix
            if (sectionId.startsWith('bible-')) {
                await API.updateBibleSection(this.currentProjectId, sectionId, value);
            } else if (sectionId.startsWith('char-')) {
                await API.updateCharacterSection(this.currentProjectId, sectionId, value);
            } else if (sectionId.startsWith('outline-')) {
                await API.updateOutlineSection(this.currentProjectId, sectionId, value);
            }
            document.getElementById('phase-label').textContent = '已保存 ✓';
            setTimeout(() => {
                const pl = document.getElementById('phase-label');
                if (pl && pl.textContent === '已保存 ✓') pl.textContent = '就绪';
            }, 2000);
        } catch (err) {
            alert(`保存失败: ${err.message}`);
        }
    },

    // ================================================================
    // Workflow
    // ================================================================
    async startWorkflow() {
        if (!this.currentProjectId) return;
        try {
            await API.startWorkflow(this.currentProjectId);
            document.getElementById('phase-label').textContent = '启动中...';
            document.getElementById('progress-fill').style.width = '2%';
        } catch (err) { alert(`启动失败: ${err.message}`); }
    },

    async confirmPhase() {
        if (!this.currentProjectId) return;
        const insp = document.getElementById('inspiration-input')?.value || '';
        try {
            await API.confirmPhase(this.currentProjectId, insp || null);
            document.querySelector('.phase-confirm')?.remove();
            document.getElementById('phase-label').textContent = '运行中...';
            document.getElementById('progress-fill').style.width = '5%';
        } catch (err) { alert(`确认失败: ${err.message}`); }
    },

    async submitScan() {
        if (!this.currentProjectId) return;
        const feiluHtml = document.getElementById('scan-feilu')?.value || '';
        const fanqieHtml = document.getElementById('scan-fanqie')?.value || '';
        if (!feiluHtml && !fanqieHtml) { alert('请至少粘贴一个平台的榜单页面内容'); return; }
        try {
            this._updatePipelineBadge('scan', 'running');
            document.getElementById('scan-input-area').style.display = 'none';
            document.getElementById('btn-start-scan').style.display = 'none';
            await API.submitScan(this.currentProjectId, feiluHtml || null, fanqieHtml || null);
            await API.startWorkflow(this.currentProjectId);
        } catch (err) { alert(`提交失败: ${err.message}`); }
    },

    /** Skip scan and go directly to bible construction */
    async skipScan() {
        if (!this.currentProjectId) return;
        this._updatePipelineBadge('scan', 'done');
        this._updatePipelineBadge('bible', 'active');
        try { await API.startWorkflow(this.currentProjectId); }
        catch (err) { alert(`启动失败: ${err.message}`); }
    },

    /** Initialize pipeline — first step is scan with "开始扫榜" button */
    _initPipeline() {
        const steps = ['scan','topic','arc','bible','chars','outline','write'];
        steps.forEach(s => { this._pipelineState[s] = 'pending'; });
        this._updatePipelineBadge('scan', 'active');
        // Show start button, hide everything else
        const btnStart = document.getElementById('btn-start-scan');
        if (btnStart) btnStart.style.display = '';
        const scanInput = document.getElementById('scan-input-area');
        if (scanInput) scanInput.style.display = 'none';
        ['scan-result-area','topic-candidates-area','topic-synopsis-cards','topic-ai-area',
         'topic-confirm-area','arc-result-area','arc-confirm-area',
         'bible-result-area','bible-confirm-area','chars-result-area','chars-confirm-area',
         'outline-result-area','outline-confirm-area','write-result-area'].forEach(id => {
            const el = document.getElementById(id); if (el) el.style.display = 'none';
        });
    },

    /** Update a pipeline step badge */
    _updatePipelineBadge(step, status) {
        this._pipelineState[step] = status;
        const stepEl = document.getElementById(`pipeline-step-${step}`);
        const badgeEl = document.getElementById(`badge-${step}`);
        if (stepEl) {
            stepEl.classList.remove('active','done','running');
            if (status === 'active' || status === 'running') stepEl.classList.add(status);
            else if (status === 'done') stepEl.classList.add('done');
        }
        if (badgeEl) {
            const labels = { scan:'平台扫榜', topic:'选题研究', arc:'小事件大纲', bible:'世界观构建', chars:'角色创建', outline:'大纲生成', write:'章节写作' };
            const statusLabels = { pending:'待执行', active:'当前步骤', running:'执行中...', done:'已完成 ✓' };
            badgeEl.textContent = statusLabels[status] || status;
            badgeEl.className = 'pipeline-badge ' + (status === 'done' ? 'done' : status === 'running' ? 'running' : status === 'active' ? 'active' : '');
        }
    },

    /** Restore pipeline state from project metadata */
    _restorePipelineState(meta) {
        const phase = meta.current_phase || 'idle';
        const phaseOrder = { idle:0, platform_scan:1, topic_selection:2, mini_arc_outline:3, bible_construction:4, character_creation:5, master_outline:6 };
        const stepMap = { 0:'scan', 1:'topic', 2:'arc', 3:'bible', 4:'chars', 5:'outline' };
        const idx = phaseOrder[phase] || 0;
        Object.keys(stepMap).forEach(k => {
            const s = stepMap[k];
            if (parseInt(k) < idx) this._updatePipelineBadge(s, 'done');
            else if (parseInt(k) === idx) this._updatePipelineBadge(s, 'active');
        });
        // If chapters have been written, activate write step
        if (meta.current_chapter > 0) this._updatePipelineBadge('write', 'active');
        // Show the start button if we're at scan step
        if (idx <= 1) {
            const btnStart = document.getElementById('btn-start-scan');
            if (btnStart) btnStart.style.display = '';
        }
        // Hide scan input unless we have scan data
        const scanInput = document.getElementById('scan-input-area');
        if (scanInput) scanInput.style.display = 'none';
    },

    /** Skip scan and go directly to bible construction */
    async skipScan() {
        if (!this.currentProjectId) return;
        this._updatePipelineBadge('scan', 'done');
        this._updatePipelineBadge('topic', 'done');
        this._updatePipelineBadge('arc', 'done');
        this._updatePipelineBadge('bible', 'active');
        document.getElementById('btn-start-scan').style.display = 'none';
        document.getElementById('scan-input-area').style.display = 'none';
        try { await API.startWorkflow(this.currentProjectId); }
        catch (err) { alert(`启动失败: ${err.message}`); }
    },

    async confirmDecision(decision) {
        if (!this.currentProjectId) return;
        const fb = document.getElementById('feedback-input')?.value || '';
        try { await API.submitDecision(this.currentProjectId, decision, fb || null, null); document.querySelector('.confirm-panel')?.remove(); }
        catch (err) { alert(`提交失败: ${err.message}`); }
    },

    toggleChapterEdit(num) {
        const display = document.getElementById(`chapter-display-${num}`);
        const editor = document.getElementById(`chapter-editor-${num}`);
        if (!display || !editor) return;
        const editing = editor.style.display !== 'none';
        if (editing) {
            editor.style.display = 'none'; display.style.display = '';
        } else {
            document.getElementById(`chapter-textarea-${num}`).value = display.innerText;
            editor.style.display = ''; display.style.display = 'none';
        }
    },

    async saveChapterEdit(num) {
        const textarea = document.getElementById(`chapter-textarea-${num}`);
        if (!textarea) return;
        const newContent = textarea.value;
        try {
            await API.updateBibleSection(this.currentProjectId, `chapter-${num}`, newContent);
            document.getElementById(`chapter-display-${num}`).innerHTML = Components._esc(newContent);
            document.getElementById(`chapter-editor-${num}`).style.display = 'none';
            document.getElementById(`chapter-display-${num}`).style.display = '';
            document.getElementById('phase-label').textContent = '已保存 ✓';
            setTimeout(() => { const pl = document.getElementById('phase-label'); if (pl && pl.textContent === '已保存 ✓') pl.textContent = '就绪'; }, 2000);
        } catch (err) { alert(`保存失败: ${err.message}`); }
    },

    async deleteChapter(num) {
        if (!this.currentProjectId || !confirm(`确认删除第${num}章？此操作不可恢复。`)) return;
        try {
            await API.deleteChapter(this.currentProjectId, num);
            this._loadTabContent('chapters');
        } catch (err) { alert(`删除失败: ${err.message}`); }
    },

    async deleteCurrentProject() {
        if (!this.currentProjectId || !confirm('确认删除？')) return;
        try {
            await API.deleteProject(this.currentProjectId);
            this.currentProjectId = null; this._workflowDone = false; if (this.ws) { this.ws.close(); this.ws = null; }
            await this._refreshProjects();
            this._navigate('my-works');
        } catch (err) { alert(`删除失败: ${err.message}`); }
    },

    // ================================================================
    // WebSocket
    // ================================================================
    _connectWS(projectId) {
        if (this.ws) this.ws.close();
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${proto}//${location.host}/ws/${projectId}`);
        this.ws.onopen = () => this._updateApiStatus(true);
        this.ws.onmessage = (e) => this._handleWSMessage(JSON.parse(e.data));
        this.ws.onclose = () => this._updateApiStatus(false);
        this.ws.onerror = () => this._updateApiStatus(false);
    },

    _handleWSMessage(msg) {
        const { type, phase, progress, message, chapter, scores, token, data: msgData } = msg;
        const pb = document.getElementById('progress-fill');
        const pl = document.getElementById('phase-label');
        const pm = { platform_scan:'平台扫榜', topic_selection:'选题研究', mini_arc_outline:'小事件大纲', bible_construction:'构建世界观', character_creation:'创建角色', master_outline:'生成大纲', chapter_planning:'规划章节', chapter_writing:'写作中', quality_review:'审阅中', polish_revision:'润色中', memory_update:'更新记忆' };
        const tm = { platform_scan:'topics', topic_selection:'topics', mini_arc_outline:'topics', bible_construction:'bible', character_creation:'characters', master_outline:'outline', chapter_planning:'chapters', chapter_writing:'chapters' };
        // Pipeline badge mapping
        const phaseToStep = { platform_scan:'scan', topic_selection:'topic', mini_arc_outline:'arc', bible_construction:'bible', character_creation:'chars', master_outline:'outline' };

        switch (type) {
            case 'phase_update':
                if (pl) pl.textContent = `${pm[phase]||phase} · ${message||''}`;
                if (pb) pb.style.width = `${(progress||0)*100}%`;
                if (tm[phase]) this._switchTab(tm[phase]);
                // Update pipeline badge to running
                if (phaseToStep[phase]) this._updatePipelineBadge(phaseToStep[phase], 'running');
                // Hide scan start button when scan starts
                if (phase === 'platform_scan') {
                    const btnStart = document.getElementById('btn-start-scan');
                    if (btnStart) btnStart.style.display = 'none';
                }
                break;
            case 'phase_complete':
                if (pl) pl.textContent = `${pm[phase]||phase} 完成 ✓`;
                if (pb) pb.style.width = '100%';
                if (phaseToStep[phase]) {
                    const step = phaseToStep[phase];
                    this._updatePipelineBadge(step, 'done');
                    if (msgData) this._renderPhaseResult(step, msgData);
                    // Show confirm UI per step
                    if (step === 'scan') {
                        // Auto-confirm scan → proceed to topic selection
                        this._updatePipelineBadge('topic', 'active');
                        API.confirmPhase(this.currentProjectId, null).catch(() => {});
                        // Clear scan status text
                        const statusEl = document.getElementById('scan-status');
                        if (statusEl) statusEl.innerHTML = '';
                    } else if (step === 'topic') {
                        document.getElementById('topic-confirm-area').style.display = 'flex';
                    } else if (step === 'arc') {
                        document.getElementById('arc-confirm-area').style.display = 'flex';
                    } else if (step === 'bible') {
                        document.getElementById('bible-confirm-area').style.display = 'flex';
                    } else if (step === 'chars') {
                        document.getElementById('chars-confirm-area').style.display = 'flex';
                    } else if (step === 'outline') {
                        document.getElementById('outline-confirm-area').style.display = 'flex';
                    }
                }
                if (tm[phase]) { this._switchTab(tm[phase]); this._loadTabContent(tm[phase]); }
                break;
            case 'phase_blocked': this._showPhaseConfirmation(phase); break;
            case 'token': if (token) this._appendStreamToken(token); else this._onStreamEnd(); break;
            case 'stream_end': this._onStreamEnd(); break;
            case 'chapter_complete':
                this._updatePipelineBadge('write', 'active');
                if (chapter && scores) {
                    const ov = document.getElementById('overview-content');
                    if (ov && !ov.querySelector('.confirm-panel')) ov.insertAdjacentHTML('beforeend', Components.confirmPanel(chapter, { overall_score:scores.overall_score||7, dimension_scores:scores, issues:[] }));
                }
                break;
            case 'workflow_complete':
                this._workflowDone = true;
                this._updatePipelineBadge('write', 'done');
                if (pl) pl.textContent = '全部完成 ✓';
                if (pb) pb.style.width = '100%';
                if (this.currentProjectId) this._loadTabContent('chapters');
                this._refreshProjects();
                break;
            case 'error': alert(`工作流错误: ${data.error}`); break;
        }
    },

    _renderPhaseResult(step, data) {
        if (!data) return;
        try {
            if (step === 'scan') this._renderScanResult(data);
            else if (step === 'topic') this._renderTopicResult(data);
            else if (step === 'arc') this._renderArcResult(data);
            else if (step === 'bible') this._renderBibleResult(data);
            else if (step === 'chars') this._renderCharsResult(data);
            else if (step === 'outline') this._renderOutlineResult(data);
            else if (step === 'write') this._renderWriteResult(data);
        } catch(e) { console.error('Render phase result error:', e); }
    },

    _renderScanResult(data) {
        const el = document.getElementById('scan-result-area');
        if (!el) return;
        let totalEntries = 0, html = '';
        const genreCount = {};
        const allEntries = [];

        const collectEntries = (platformData, platformName) => {
            if (!platformData?.entries) return;
            (platformData.entries || []).forEach(e => {
                if (e.genre) {
                    const g = e.genre.replace(/[\/／].*/, '').trim();
                    genreCount[g] = (genreCount[g] || 0) + 1;
                }
                totalEntries++;
                allEntries.push({...e, _platform: platformName});
            });
        };
        collectEntries(data.feilu, '飞卢');
        collectEntries(data.fanqie, '番茄');

        // Dashboard tiles
        html += '<div class="dash-grid">';
        html += `<div class="dash-tile accent"><div class="dash-num">${totalEntries}</div><div class="dash-label">扫描书籍</div></div>`;
        const platforms = [data.feilu, data.fanqie].filter(Boolean);
        html += `<div class="dash-tile success"><div class="dash-num">${platforms.length}</div><div class="dash-label">覆盖平台</div></div>`;
        html += `<div class="dash-tile warning"><div class="dash-num">${Object.keys(genreCount).length}</div><div class="dash-label">识别题材</div></div>`;
        html += '</div>';

        // Bar chart of top genres
        const sortedGenres = Object.entries(genreCount).sort((a,b) => b[1]-a[1]).slice(0, 8);
        if (sortedGenres.length) {
            const maxCount = sortedGenres[0][1];
            html += '<p style="font-weight:600;font-size:0.82rem;margin-bottom:6px">📊 热门题材分布</p>';
            html += '<div class="bar-chart">';
            sortedGenres.forEach(([g, c], i) => {
                const pct = Math.round((c/maxCount)*100);
                const colors = ['','warm','cool','green'];
                html += `<div class="bar-row"><span class="bar-label">${Components._esc(g)}</span><div class="bar-track"><div class="bar-fill ${colors[i%4]||''}" style="width:${pct}%"></div></div><span class="bar-val">${c}</span></div>`;
            });
            html += '</div>';
        }

        // Book list by platform
        platforms.forEach(p => {
            const isFeilu = p.platform === '飞卢' || data.feilu === p;
            const entries = p.entries || [];
            html += `<div class="scan-mini-card" style="margin-bottom:10px"><h4>${isFeilu?'🔍 飞卢':'🍅 番茄'} · ${Components._esc(p.list_name||'榜单')} · ${Components._esc(p.date||'')}</h4>`;
            if (p.scan_failed) {
                html += `<p style="color:var(--error)">${Components._esc(p.summary||'扫榜失败')}</p>`;
            } else {
                html += `<p style="font-size:0.78rem;color:var(--text-secondary);margin-bottom:8px">${Components._esc((p.summary||'').substring(0,120))}</p>`;
                if (entries.length) {
                    html += '<div class="scan-book-list">';
                    entries.forEach((e, i) => {
                        const title = e.title || '';
                        html += `<div class="scan-book-row"><span class="scan-book-rank">#${i+1}</span><span class="scan-book-title">${Components._esc(title)}</span>${e.genre?`<span class="tag-chip gray">${Components._esc(e.genre)}</span>`:''}</div>`;
                    });
                    html += '</div>';
                }
            }
            html += '</div>';
        });

        el.innerHTML = html; el.style.display = 'block';
        document.getElementById('scan-input-area').style.display = 'none';
        document.getElementById('btn-start-scan').style.display = 'none';
        // Clear status
        const statusEl = document.getElementById('scan-status');
        if (statusEl) statusEl.innerHTML = '';
    },

    _renderTopicResult(data) {
        this._topicScores = data.scores?.scores || [];
        this._topicTitleData = data.title_synopsis || [];
        this._topicCandidates = data.candidates?.topics || [];
        this._topicCrossPlatform = data.cross_platform;
        this._checkedCandidates = {};
        this._synopsisCards = [];
        this._selectedSynopsisIdx = -1;

        const allTopics = [];
        const seen = new Set();

        // Priority 1: Scored topics
        for (const t of this._topicScores) {
            const name = t.genre_name;
            if (name && !seen.has(name)) { seen.add(name); allTopics.push({name, score: t.total_score, data: t}); }
        }
        // Priority 2: Candidates without scores
        for (const t of this._topicCandidates) {
            const name = t.genre_name;
            if (name && !seen.has(name)) { seen.add(name); allTopics.push({name, score: 0, data: t}); }
        }

        const candidatesArea = document.getElementById('topic-candidates-area');
        const candidatesList = document.getElementById('topic-candidates-list');
        const aiArea = document.getElementById('topic-ai-area');

        if (!allTopics.length) {
            // No topics at all — show retry UI with helpful context
            if (candidatesArea) candidatesArea.style.display = 'none';
            if (aiArea) {
                aiArea.style.display = 'block';
                const scanOk = (data.feilu?.entries?.length || 0) + (data.fanqie?.entries?.length || 0) > 0;
                const msg = scanOk
                    ? '⚠️ 扫榜已完成但未成功生成候选题材，可能是AI生成步骤出现问题。请点击重试，或在下方输入灵感手动生成。'
                    : '⚠️ 未生成候选题材，可能扫榜数据不足。请点击重试按钮，或在下方输入你的灵感。';
                aiArea.querySelector('p').innerHTML = msg +
                    '&nbsp;<button class="btn btn-warning btn-sm" onclick="App.retryPhase(\'topic_selection\')" style="margin-left:8px">🔄 重新生成选题</button>';
                aiArea.querySelector('p').style.color = 'var(--warning)';
            }
            return;
        }

        // Show candidates
        if (candidatesArea && candidatesList) {
            candidatesArea.style.display = 'block';
            let listHtml = '';
            allTopics.slice(0, 12).forEach((t, i) => {
                const name = Components._esc(t.name);
                const score = t.score ? (t.score).toFixed(1) : '—';
                listHtml += `<div class="topic-candidate-row" data-genre="${name}" onclick="App.toggleCandidate('${name}')">
                    <div class="topic-candidate-check">✓</div>
                    <span class="topic-candidate-name">${name}</span>
                    <span class="topic-candidate-score">${score}${t.score?'分':''}</span>
                </div>`;
            });
            candidatesList.innerHTML = listHtml;
            this._allTopics = allTopics.slice(0, 12);
        }

        if (aiArea) aiArea.style.display = 'block';
    },

    /** Toggle candidate checkbox */
    toggleCandidate(genreName) {
        this._checkedCandidates[genreName] = !this._checkedCandidates[genreName];
        const row = document.querySelector(`.topic-candidate-row[data-genre="${genreName}"]`);
        if (row) row.classList.toggle('checked', this._checkedCandidates[genreName]);
        const count = Object.values(this._checkedCandidates).filter(Boolean).length;
        const hint = document.getElementById('candidate-hint');
        if (hint) hint.textContent = `已选 ${count} 个`;
    },

    /** Generate synopsis cards for checked candidates, calling LLM on demand */
    async generateSynopses() {
        const checked = Object.entries(this._checkedCandidates).filter(([_,v]) => v).map(([k]) => k);
        if (!checked.length) { alert('请至少勾选一个题材'); return; }

        const cardsContainer = document.getElementById('topic-synopsis-cards');
        if (!cardsContainer) return;
        cardsContainer.style.display = 'block';
        cardsContainer.innerHTML = '<p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:8px">📝 正在生成书名和简介...</p>';

        this._synopsisCards = [];
        this._selectedSynopsisIdx = -1;

        const titleData = this._topicTitleData || [];
        const inspEl = document.getElementById('topic-ai-input');
        const inspiration = inspEl?.value?.trim() || '';

        for (let idx = 0; idx < checked.length; idx++) {
            const genreName = checked[idx];
            const ts = titleData.find(t => t.genre_name === genreName);
            let title = ts?.final_title || '';
            let synopsis = ts?.final_synopsis || '';

            // If no pre-generated synopsis, call API to generate on demand
            if (!title || !synopsis) {
                try {
                    cardsContainer.innerHTML = `<p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:8px">📝 正在为「${Components._esc(genreName)}」生成书名和简介...</p>`;
                    const result = await API.generateSynopsis(this.currentProjectId, genreName, inspiration);
                    title = result.final_title || genreName;
                    synopsis = result.final_synopsis || '';
                } catch (err) {
                    console.error('Synopsis generation failed for', genreName, err);
                    title = title || genreName;
                    synopsis = synopsis || inspiration || '';
                }
            }

            const card = { genreName, title, synopsis, editedTitle: title, editedSynopsis: synopsis };
            this._synopsisCards.push(card);
            const cardIdx = this._synopsisCards.length - 1;
            const cardHtml = this._renderSynopsisCard(card, cardIdx);
            if (idx === 0) {
                cardsContainer.innerHTML = '<p style="font-weight:600;font-size:0.82rem;margin-bottom:6px">📝 书名和简介（点击圆圈选择，标题和简介可编辑）</p>';
            }
            cardsContainer.insertAdjacentHTML('beforeend', cardHtml);
        }

        // Show confirm bar after all cards generated
        document.getElementById('topic-confirm-area').style.display = 'flex';
    },

    /** Render a single synopsis card */
    _renderSynopsisCard(card, idx) {
        const name = Components._esc(card.genreName);
        return `<div class="synopsis-card" id="synopsis-card-${idx}">
            <div class="synopsis-card-header">
                <div class="synopsis-radio" onclick="App.selectSynopsisCard(${idx})" title="选择这个书名"></div>
                <input class="synopsis-title-input" id="synopsis-title-${idx}" value="${Components._esc(card.editedTitle)}" onchange="App._synopsisCards[${idx}].editedTitle=this.value">
            </div>
            <div class="synopsis-body">
                <textarea class="synopsis-desc-input" id="synopsis-desc-${idx}" onchange="App._synopsisCards[${idx}].editedSynopsis=this.value">${Components._esc(card.editedSynopsis)}</textarea>
            </div>
            <div class="synopsis-meta">
                <span>题材：${name}</span>
            </div>
        </div>`;
    },

    /** Radio-select a synopsis card */
    selectSynopsisCard(idx) {
        this._selectedSynopsisIdx = idx;
        document.querySelectorAll('.synopsis-card').forEach((el, i) => {
            el.classList.toggle('selected', i === idx);
        });
    },

    /** AI generate titles based on user inspiration */
    async aiGenerateTitles() {
        const input = document.getElementById('topic-ai-input');
        const resultsEl = document.getElementById('topic-ai-results');
        const btn = document.getElementById('btn-ai-generate');
        const insp = input?.value?.trim();
        if (!insp) { alert('请先输入灵感或想法'); return; }

        btn.textContent = '生成中...'; btn.disabled = true;
        resultsEl.innerHTML = '<span style="color:var(--text-secondary);font-size:0.82rem">AI 正在分析榜单数据并生成书名...</span>';

        try {
            const r = await fetch('/api/projects/' + this.currentProjectId + '/ai-titles', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({inspiration: insp})
            });
            if (!r.ok) throw new Error((await r.json()).detail || '请求失败');
            const data = await r.json();
            if (data.titles?.length) {
                resultsEl.innerHTML = '<p style="font-size:0.78rem;color:var(--text-secondary);margin-bottom:4px">AI 生成的书名（点击添加到候选列表）：</p>';
                data.titles.forEach(t => {
                    const chip = document.createElement('span');
                    chip.className = 'tag-chip purple';
                    chip.style.cssText = 'cursor:pointer;margin:3px;font-size:0.82rem;padding:4px 12px';
                    chip.textContent = t;
                    chip.onclick = () => {
                        input.value = t;
                        // Add to checked candidates
                        const name = t;
                        if (!this._checkedCandidates[name]) {
                            this._checkedCandidates[name] = true;
                            const row = document.querySelector(`.topic-candidate-row[data-genre="${Components._esc(name)}"]`);
                            if (row) row.classList.add('checked');
                            else {
                                // Add a new row
                                this._allTopics.push({name, score: 0, data: {}});
                                const list = document.getElementById('topic-candidates-list');
                                const newRow = document.createElement('div');
                                newRow.className = 'topic-candidate-row checked';
                                newRow.dataset.genre = name;
                                newRow.innerHTML = `<div class="topic-candidate-check">✓</div><span class="topic-candidate-name">${Components._esc(name)}</span><span class="topic-candidate-score">AI</span>`;
                                newRow.onclick = () => App.toggleCandidate(name);
                                list.appendChild(newRow);
                            }
                            const count = Object.values(this._checkedCandidates).filter(Boolean).length;
                            const hint = document.getElementById('candidate-hint');
                            if (hint) hint.textContent = `已选 ${count} 个`;
                        }
                    };
                    resultsEl.appendChild(chip);
                });
            } else {
                resultsEl.innerHTML = '<span style="color:var(--text-secondary);font-size:0.82rem">未生成结果，请尝试更具体的灵感描述</span>';
            }
        } catch (err) {
            resultsEl.innerHTML = `<span style="color:var(--error);font-size:0.82rem">生成失败: ${err.message}</span>`;
        } finally {
            btn.textContent = '生成'; btn.disabled = false;
        }
    },

    _renderArcResult(data) {
        const el = document.getElementById('arc-result-area');
        if (!el) return;
        const outlines = data.mini_arc_outline || data;
        const keys = Object.keys(outlines).filter(k => k !== 'mini_arc_outline');
        // Filter entries that actually have chapter data (not just empty array)
        const entries = keys.map(k => [k, outlines[k]]).filter(([_,v]) => v?.chapters?.length > 0);
        let html = '';

        if (!entries.length) {
            el.innerHTML = `<div style="padding:16px;border:1px dashed var(--warning);border-radius:8px;text-align:center">
                <p style="color:var(--warning);margin-bottom:12px">⚠️ 大纲生成未完成（可能是模型结构化输出失败）</p>
                <p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:12px">请在下方输入框中补充更多细节，然后点击重新生成</p>
                <button class="btn btn-warning" onclick="App.retryPhase('mini_arc_outline')" style="margin-right:8px">🔄 重新生成大纲</button>
            </div>`;
            el.style.display = 'block';
            // Show the confirm/inspiration area so user can add details
            const confirmEl = document.getElementById('arc-confirm-area');
            if (confirmEl) confirmEl.style.display = 'flex';
            return;
        }

        // Stats
        const totalChapters = entries.reduce((sum, [_,v]) => sum + (v.chapters?.length||0), 0);
        html += '<div class="dash-grid">';
        html += `<div class="dash-tile accent"><div class="dash-num">${entries.length}</div><div class="dash-label">生成方案</div></div>`;
        html += `<div class="dash-tile success"><div class="dash-num">${totalChapters}</div><div class="dash-label">总章节数</div></div>`;
        html += `<div class="dash-tile warning"><div class="dash-num">2万</div><div class="dash-label">单方案字数</div></div>`;
        html += '</div>';

        entries.forEach(([genre, arc]) => {
            html += `<div style="margin-bottom:12px"><p style="font-weight:600;margin-bottom:6px">📋 ${Components._esc(genre)} · ${Components._esc(arc.total_words||'约2万字')}</p>`;
            html += '<div class="timeline">';
            (arc.chapters||[]).slice(0, 10).forEach((c, i) => {
                html += `<div class="tl-node${i<3?' done':''}"><div class="tl-card"><div class="tl-title">第${c.chapter_number||(i+1)}章</div><div class="tl-sub">${Components._esc((c.goal||c.conflict||'').substring(0,50))}</div></div></div>`;
            });
            html += '</div></div>';
        });

        el.innerHTML = html; el.style.display = 'block';
    },

    _renderBibleResult(data) {
        const el = document.getElementById('bible-result-area');
        if (!el) return;
        const w = data.world || {};
        const factions = data.factions || [];
        const themes = data.themes || [];
        const style = data.style_contract || {};
        let html = '';

        // Dashboard tiles
        html += '<div class="dash-grid">';
        html += `<div class="dash-tile accent"><div class="dash-num">${w.name ? 1 : 0}</div><div class="dash-label">世界观</div></div>`;
        html += `<div class="dash-tile success"><div class="dash-num">${factions.length}</div><div class="dash-label">势力组织</div></div>`;
        html += `<div class="dash-tile warning"><div class="dash-num">${themes.length}</div><div class="dash-label">核心主题</div></div>`;
        html += `<div class="dash-tile accent"><div class="dash-num">${w.magic_system?'✓':'—'}</div><div class="dash-label">力量体系</div></div>`;
        html += '</div>';

        // World summary
        if (w.name) {
            html += `<div style="margin-bottom:8px"><strong>${Components._esc(w.name)}</strong> <span class="tag-chip blue">${Components._esc(w.world_type||'')}</span></div>`;
            if (w.geography) html += `<p style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">🌏 ${Components._esc(w.geography.substring(0,100))}</p>`;
            if (w.culture) html += `<p style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">🏛 ${Components._esc(w.culture.substring(0,100))}</p>`;
            if (w.technology_level) html += `<p style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:4px">⚙ ${Components._esc(w.technology_level)}</p>`;
        }

        // Faction chips
        if (factions.length) {
            html += '<p style="font-weight:600;font-size:0.8rem;margin-top:8px">势力：</p>';
            factions.forEach(f => {
                html += `<span class="tag-chip purple">${Components._esc(f.name)}</span>`;
            });
            html += '<br>';
        }

        // Style contract summary
        if (style.tone) {
            html += '<p style="font-size:0.78rem;color:var(--text-secondary);margin-top:8px">文风：';
            html += `<span class="tag-chip gray">${Components._L(Components.TONE, style.tone)}</span>`;
            if (style.pacing_preference) html += `<span class="tag-chip gray">${Components._L(Components.PACING, style.pacing_preference)}</span>`;
            if (style.sentence_style) html += `<span class="tag-chip gray">${Components._L(Components.SENTENCE, style.sentence_style)}</span>`;
            html += '</p>';
        }

        el.innerHTML = html; el.style.display = 'block';
    },

    _renderCharsResult(data) {
        const el = document.getElementById('chars-result-area');
        if (!el) return;
        const chars = data.characters || {};
        const list = Object.values(chars);
        let html = '';

        // Stats
        html += '<div class="dash-grid">';
        html += `<div class="dash-tile accent"><div class="dash-num">${list.length}</div><div class="dash-label">角色总数</div></div>`;
        const roles = { protagonist:0, antagonist:0, deuteragonist:0, supporting:0, mentor:0 };
        list.forEach(c => { if (roles[c.role] !== undefined) roles[c.role]++; });
        html += `<div class="dash-tile success"><div class="dash-num">${roles.protagonist||0}</div><div class="dash-label">主角</div></div>`;
        html += `<div class="dash-tile warning"><div class="dash-num">${roles.antagonist||0}</div><div class="dash-label">反派</div></div>`;
        html += '</div>';

        // Character cards
        list.forEach(c => {
            const roleLabel = Components._L(Components.ROLE, c.role) || '';
            const archetypeLabel = Components._L(Components.ARCHETYPE, c.archetype) || '';
            html += `<div class="char-card">
                <span class="char-name">${Components._esc(c.name)}</span>
                ${roleLabel?`<span class="tag-chip purple">${roleLabel}</span>`:''}
                ${archetypeLabel?`<span class="tag-chip blue">${archetypeLabel}</span>`:''}
                <p style="font-size:0.8rem;margin-top:4px;color:var(--text-secondary)">${Components._esc((c.personality||'').substring(0,80))}</p>
                ${c.motivation?`<p style="font-size:0.78rem;color:var(--text-secondary)">动机：${Components._esc(c.motivation.substring(0,60))}</p>`:''}
            </div>`;
        });

        el.innerHTML = html; el.style.display = 'block';
    },

    _renderOutlineResult(data) {
        const el = document.getElementById('outline-result-area');
        if (!el) return;
        let html = '';
        const volumes = data.volumes || [];
        const mainPlot = data.main_plot || [];
        const subplots = data.subplots || [];
        const tps = data.major_turning_points || [];

        // Stats
        html += '<div class="dash-grid">';
        html += `<div class="dash-tile accent"><div class="dash-num">${volumes.length}</div><div class="dash-label">分卷</div></div>`;
        html += `<div class="dash-tile success"><div class="dash-num">${data.chapter_count||'?'}</div><div class="dash-label">计划章节</div></div>`;
        html += `<div class="dash-tile warning"><div class="dash-num">${mainPlot.length+subplots.length}</div><div class="dash-label">故事线</div></div>`;
        html += '</div>';

        // Logline
        if (data.logline) {
            html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:10px;font-style:italic">"${Components._esc(data.logline.substring(0,120))}"</p>`;
        }

        // Volume tree
        if (volumes.length) {
            html += '<div class="volume-tree">';
            volumes.forEach(v => {
                html += `<div class="volume-node">
                    <div class="volume-header"><span class="vol-num">${v.number||'?'}</span>${Components._esc(v.title||'未命名')}</div>
                    <div class="volume-body">第${v.start_chapter||'?'}-${v.end_chapter||'?'}章 · ${Components._esc(v.logline||'')}</div>
                </div>`;
            });
            html += '</div>';
        }

        // Turning points timeline
        if (tps.length) {
            html += '<p style="font-weight:600;font-size:0.8rem;margin-top:10px">⏳ 关键转折点</p>';
            html += '<div class="timeline">';
            tps.forEach(tp => {
                html += `<div class="tl-node done"><div class="tl-card"><div class="tl-title">第${tp.chapter||'?'}章：${Components._L(Components.TURNING, tp.turning_type)}</div><div class="tl-sub">${Components._esc((tp.description||'').substring(0,80))}</div></div></div>`;
            });
            html += '</div>';
        }

        el.innerHTML = html; el.style.display = 'block';
    },

    _renderWriteResult(data) {
        const el = document.getElementById('write-result-area');
        if (!el) return;
        let html = '';
        // Fetch chapters via API for up-to-date data
        if (this.currentProjectId) {
            API.getChapters(this.currentProjectId).then(chapters => {
                const list = Array.isArray(chapters) ? chapters : [];
                const done = list.filter(c => c.status === 'done').length;
                html += '<div class="dash-grid">';
                html += `<div class="dash-tile accent"><div class="dash-num">${done}</div><div class="dash-label">已完成</div></div>`;
                html += `<div class="dash-tile success"><div class="dash-num">${list.length}</div><div class="dash-label">总章节</div></div>`;
                html += '</div>';
                if (list.length) {
                    html += '<div class="chapter-progress-grid">';
                    list.forEach(c => {
                        const isDone = c.status === 'done';
                        html += `<div class="chapter-progress-chip ${isDone?'done':'active'}" title="第${c.chapter_number}章">${c.chapter_number}</div>`;
                    });
                    html += '</div>';
                }
                el.innerHTML = html; el.style.display = 'block';
            }).catch(() => {
                el.innerHTML = '<p style="font-size:0.85rem;color:var(--text-secondary)">章节生成进行中...</p>';
                el.style.display = 'block';
            });
        }
    },

    _appendStreamToken(token) {
        let el = document.getElementById('stream-output');
        if (!el) {
            const at = document.querySelector('#view-project .tab-content.active');
            if (!at) return;
            const c = document.createElement('div'); c.className = 'stream-container';
            c.innerHTML = '<div class="stream-header">正在生成...</div><pre id="stream-output" class="stream-text streaming"></pre>';
            at.insertBefore(c, at.firstChild); el = document.getElementById('stream-output');
        }
        if (el) { el.textContent += token; el.scrollTop = el.scrollHeight; }
    },

    _onStreamEnd() {
        document.getElementById('stream-output')?.classList.remove('streaming');
        setTimeout(() => { const at = document.querySelector('#view-project .tab.active'); if (at) this._loadTabContent(at.dataset.tab); }, 500);
    },

    _showPhaseConfirmation(phase) {
        document.querySelector('.phase-confirm')?.remove();
        const ac = document.querySelector('#view-project .tab-content.active');
        if (!ac) return;
        ac.insertAdjacentHTML('beforeend', Components.phaseConfirmPanel(phase));
    },

    _switchTab(n) { document.querySelector(`#view-project .tab[data-tab="${n}"]`)?.click(); },

    // ================================================================
    // 智能写作助手
    // ================================================================
    async handleAssistantFile(event) {
        const file = event.target.files[0]; if (!file) return;
        const fd = new FormData(); fd.append('file', file);
        try {
            const r = await fetch('/api/projects/default/editor/upload', { method:'POST', body:fd });
            if (!r.ok) throw new Error((await r.json()).detail);
            const d = await r.json(); this._assistantContext = d.file_id;
            document.getElementById('assistant-file-name').textContent = `已上传: ${d.filename}`;
            this._addChatMsg('assistant', `已上传「${d.filename}」，共 ${d.size.toLocaleString()} 字。\n\n预览：\n${d.preview}`);
        } catch (err) { alert(`上传失败: ${err.message}`); }
    },

    quickChat(msg) { document.getElementById('chat-input').value = msg; this.sendAssistantMessage(); },

    async sendAssistantMessage() {
        const input = document.getElementById('chat-input'); const msg = input.value.trim(); if (!msg) return;
        const mode = document.getElementById('chat-mode')?.value || 'chat'; input.value = '';
        this._addChatMsg('user', msg);
        const div = this._addChatMsg('assistant', ''); const td = div.querySelector('.chat-text');
        try {
            const r = await fetch('/api/projects/default/editor/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ message:msg, mode, context:this._assistantContext||null }) });
            if (!r.ok) throw new Error((await r.json()).detail);
            const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
            while (true) {
                const { done, value } = await reader.read(); if (done) break;
                buf += dec.decode(value, { stream:true });
                const lines = buf.split('\n\n'); buf = lines.pop() || '';
                for (const l of lines) { if (l.startsWith('data: ')) { const t = l.slice(6); if (t==='[DONE]') return; if (t.startsWith('[ERROR]')) { td.textContent += t.slice(8); return; } td.textContent += t; } }
            }
        } catch (err) { td.textContent = `请求失败: ${err.message}`; }
    },

    _addChatMsg(role, content) {
        const msgs = document.getElementById('chat-messages'); const w = msgs.querySelector('.chat-welcome'); if (w) w.remove();
        const d = document.createElement('div'); d.className = `chat-msg ${role}`;
        d.innerHTML = `<div class="chat-text">${content||''}</div>`;
        msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d;
    },

    // ================================================================
    // New Project Form
    // ================================================================
    // Form handlers removed — project creation is now automatic on "开始扫榜"
    _setupFormHandlers() {
        // No setup needed — the old modal form has been removed.
        // Projects are auto-created when clicking "开始扫榜" and finalized on topic confirmation.
    },

    _updateApiStatus(c) {
        const el = document.getElementById('api-status'); if (!el) return;
        if (c===undefined) { fetch('/api/projects').then(()=>{el.className='status-dot online';}).catch(()=>{el.className='status-dot offline';}); }
        else { el.className = c ? 'status-dot online' : 'status-dot offline'; }
    },
    _esc(s) { return Components._esc(s); },

    // ================================================================
    // Scan Step — auto-creates draft project, auto-scrapes platforms
    // ================================================================
    async startScan() {
        // 1. Auto-create draft project if needed
        if (!this.currentProjectId) {
            const draftConfig = {
                title: '未命名项目',
                inspiration: '待定',
                genre: ['待定'],
                target_readers: '网文读者',
                tone: '',
                target_length: 'novel',
                target_word_count: 80000,
                style_reference: null,
                taboo_content: [],
                language: 'zh-CN',
                special_requirements: [],
            };
            try {
                const result = await API.createProject(draftConfig);
                await this._refreshProjects();
                this.currentProjectId = result.project_id;
                this._connectWS(result.project_id);
                this._initPipeline();
                this._showProjectHeader('未命名项目');
            } catch (err) { alert(`创建失败: ${err.message}`); return; }
        }

        // 2. Hide start button, show scanning status
        document.getElementById('btn-start-scan').style.display = 'none';
        const scanBody = document.getElementById('body-scan');
        let statusEl = document.getElementById('scan-status');
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'scan-status';
            statusEl.style.cssText = 'padding:12px 0;font-size:0.9rem;color:var(--text-secondary);';
            scanBody.appendChild(statusEl);
        }
        statusEl.innerHTML = '🔍 正在自动抓取飞卢和番茄榜单数据... <span style="color:var(--accent)">请稍候</span>';

        // 3. Try auto-scan
        try {
            const result = await API.autoScan(this.currentProjectId);

            if (result.status === 'ok' || result.status === 'partial') {
                // Scan succeeded (at least partially) — start workflow
                statusEl.innerHTML = `📊 ${result.message}（飞卢:${result.feilu.chars}字符, 番茄:${result.fanqie.chars}字符）`;
                await API.startWorkflow(this.currentProjectId);
                this._updatePipelineBadge('scan', 'running');
                statusEl.innerHTML += '<br><span style="color:var(--success)">✅ 扫榜任务已提交，正在分析中...</span>';
                return;
            } else {
                // Both platforms failed
                statusEl.innerHTML = `⚠️ ${result.message}<br><small style="color:var(--text-secondary)">飞卢和番茄榜单均抓取失败，请手动粘贴页面内容</small>`;
            }
        } catch (err) {
            console.error('Auto-scan failed:', err);
            statusEl.innerHTML = '⚠️ 自动扫榜失败，请手动粘贴榜单内容';
        }

        // 4. Fallback: show manual paste area
        document.getElementById('scan-input-area').style.display = 'block';
    },

    /** Show project header in workspace */
    _showProjectHeader(title) {
        document.getElementById('workspace-default-title').style.display = 'none';
        document.getElementById('workspace-project-header').style.display = 'flex';
        document.getElementById('project-title-display').textContent = '《' + title + '》';
    },

    /** Navigate to 我的书库 */
    goToLibrary() {
        this._navigate('my-works');
    },

    // ================================================================
    // Topic Selection
    // ================================================================
    _scanProjectId: null,  // Original project that holds scan data
    _selectedTopic: null,

    selectTopic(genreName) {
        this._selectedTopic = genreName;
        document.querySelectorAll('.topic-select-item').forEach(el => {
            el.classList.toggle('selected', el.dataset.genre === genreName);
        });
        // Auto-fill the editable title
        const titleData = this._topicTitleData?.find(t => t.genre_name === genreName);
        const titleInput = document.getElementById('topic-edit-title');
        if (titleInput) {
            titleInput.value = titleData?.final_title || genreName;
        }
    },

    async confirmTopic() {
        // Get selected synopsis card or fall back to checked topic
        let newTitle, genreName;
        const insp = document.getElementById('topic-inspiration')?.value || '';

        if (this._selectedSynopsisIdx >= 0 && this._synopsisCards[this._selectedSynopsisIdx]) {
            const card = this._synopsisCards[this._selectedSynopsisIdx];
            newTitle = card.editedTitle || card.title;
            genreName = card.genreName;
        } else {
            // No synopsis card selected — use first checked candidate
            const checked = Object.entries(this._checkedCandidates || {}).filter(([_,v]) => v).map(([k]) => k);
            if (!checked.length) { alert('请先勾选题材并生成简介，然后选择一个书名'); return; }
            genreName = checked[0];
            newTitle = genreName;
        }

        try {
            await API.updateTitle(this.currentProjectId, newTitle);
            await this._refreshProjects();
            this._showProjectHeader(newTitle);

            document.getElementById('topic-confirm-area').style.display = 'none';
            this._updatePipelineBadge('topic', 'done');
            this._updatePipelineBadge('arc', 'active');

            const confirmMsg = `选定题材：${genreName}。书名：《${newTitle}》。${insp}`;
            await API.confirmPhase(this.currentProjectId, confirmMsg);
        } catch (err) { alert(`确认失败: ${err.message}`); }
    },

    // ================================================================
    // Generic Step Confirmation (arc, bible, chars, outline)
    // ================================================================
    async confirmStep(step) {
        if (!this.currentProjectId) return;
        const inspEl = document.getElementById(`${step}-inspiration`);
        const insp = inspEl?.value || '';
        const nextMap = { arc:'bible', bible:'chars', chars:'outline', outline:'write' };
        try {
            await API.confirmPhase(this.currentProjectId, insp || null);
            this._updatePipelineBadge(step, 'done');
            if (nextMap[step]) this._updatePipelineBadge(nextMap[step], 'active');
            // Hide confirm bar
            const confirmEl = document.getElementById(`${step}-confirm-area`);
            if (confirmEl) confirmEl.style.display = 'none';
        } catch (err) { alert(`确认失败: ${err.message}`); }
    },

    /** Retry a failed phase (e.g. mini_arc_outline). */
    async retryPhase(phase) {
        if (!this.currentProjectId) return;
        // Handle topic_selection UI reset
        if (phase === 'topic_selection') {
            document.getElementById('topic-candidates-area').style.display = 'none';
            document.getElementById('topic-synopsis-cards').style.display = 'none';
            document.getElementById('topic-confirm-area').style.display = 'none';
            const aiArea = document.getElementById('topic-ai-area');
            if (aiArea) {
                aiArea.style.display = 'block';
                aiArea.querySelector('p').innerHTML = '<span style="color:var(--accent)">🔄 正在重新生成候选题材，这可能需要1-2分钟...</span>';
            }
            this._updatePipelineBadge('topic', 'running');
        }
        const resultEl = document.getElementById(`${phase === 'mini_arc_outline' ? 'arc' : phase}-result-area`);
        if (resultEl) resultEl.innerHTML = '<p style="color:var(--accent)">🔄 正在重新生成...</p>';
        // Pick up any inspiration the user added
        const inspEl = document.getElementById(`${phase === 'mini_arc_outline' ? 'arc' : phase}-inspiration`);
        const insp = inspEl?.value?.trim() || '';
        try {
            await API.retryPhase(this.currentProjectId, phase, insp || null);
        } catch (err) {
            alert(`重试失败: ${err.message}`);
        }
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
