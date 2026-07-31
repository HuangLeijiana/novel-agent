/**
 * UI component renderers — clean Chinese display, no raw JSON/enum markers
 */
const Components = {
    ROLE: { protagonist:'主角', antagonist:'反派', deuteragonist:'副主角', supporting:'配角', minor:'次要角色', mentor:'导师', love_interest:'恋人', rival:'对手', foil:'镜像角色' },
    CONFLICT: {
        person_vs_person:'人与人', person_vs_society:'人与社会', person_vs_self:'人与自我',
        person_vs_nature:'人与自然', person_vs_technology:'人与科技', person_vs_faction:'势力对抗',
        person_vs_organisation:'势力对抗', person_vs_organization:'势力对抗',
        person_vs_destiny:'人与命运', person_vs_fate:'人与命运',
    },
    TURNING: { inciting_incident:'触发事件', first_plot_point:'第一转折点', midpoint:'中点转折', all_is_lost:'至暗时刻', dark_night_of_soul:'灵魂黑夜', climax:'高潮', denouement:'结局' },
    SEVERITY: { critical:'严重', major:'重要', minor:'次要', suggestion:'建议' },
    ARC: { main:'主线', subplot:'支线', b_plot:'副线' },
    TONE: { dark:'暗黑', light:'轻松', gritty:'冷硬', whimsical:'奇幻', epic:'史诗' },
    SENTENCE: { varied:'多样', simple:'简洁', ornate:'华丽', minimalist:'极简', complex:'复杂', balanced:'均衡' },
    PACING: { fast:'快节奏', medium:'中等', slow:'慢节奏', 'medium-slow':'中慢', 'medium-fast':'中快', variable:'变化' },
    NARRATIVE: { close:'沉浸式', distant:'观察式', omniscient:'全知', 'third_person_limited':'第三人称有限' },
    ARCHETYPE: { hero:'英雄', mentor:'导师', herald:'信使', trickster:'骗徒', shadow:'黑暗面', 'threshold_guardian':'守护者', shapeshifter:'千面人', ally:'盟友', 'anti_hero':'反英雄', villain:'反派', sage:'智者', explorer:'探索者', rebel:'反抗者', lover:'情人', creator:'创造者', ruler:'统治者', everyman:'普通人', caregiver:'照料者', innocent:'纯真者', jester:'小丑', orphan:'孤儿', warrior:'战士', magician:'魔法师', seeker:'追寻者', destroyer:'毁灭者' },

    /** Universal label lookup: try map first, then humanize snake_case, else return as-is */
    _L(map, key) {
        if (!key) return '';
        if (map[key]) return map[key];
        // Try common variations
        const lower = String(key).toLowerCase().replace(/[ -]/g, '_');
        if (map[lower]) return map[lower];
        // Auto-humanize: person_vs_organisation → Person Vs Organisation
        if (/^[a-z][a-z_]+$/.test(key)) {
            return String(key).split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }
        return key;
    },

    _cleanJson(s) {
        if (!s || !s.startsWith('{')) return this._esc(s||'');
        try { const o = JSON.parse(s); return [o.name, o.rules, o.description, o.power_progression ? '进阶: '+o.power_progression : ''].filter(Boolean).map(x=>this._esc(x)).join('<br>'); }
        catch(e) { return this._esc(s); }
    },

    /** Wraps a section with an edit/save toggle */
    _editableSection(id, title, contentHtml, textValue) {
        return `<div class="section editable-section" id="${id}-section">
            <div class="section-header"><h4>${title}</h4><button class="btn btn-secondary btn-sm" id="${id}-edit-btn" onclick="App.toggleEdit('${id}')">编辑</button></div>
            <div id="${id}-display">${contentHtml}</div>
            <div id="${id}-editor" style="display:none"><textarea style="width:100%;min-height:120px;padding:10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);font-family:inherit;font-size:0.9rem;line-height:1.6;resize:vertical">${this._esc(textValue||'')}</textarea></div>
        </div>`;
    },

    overviewEditable(p) {
        if (!p?.config) return '<p>加载中...</p>';
        const c = p.config;
        const content = `<p>书名：${this._esc(c.title||'未设置')}</p>
            <p>灵感：${this._esc(c.inspiration||'')}</p>
            <p>题材：${(c.genre||[]).join('、')}</p>
            <p>目标读者：${this._esc(c.target_readers||'')}</p>
            <p>语调：${this.TONE[c.tone]||c.tone||'未指定'}</p>
            <p>篇幅：${c.target_length||'未指定'}（约${(c.target_word_count||0).toLocaleString()}字）</p>
            ${c.style_reference?`<p>风格参考：${this._esc(c.style_reference)}</p>`:''}`;
        return this._editableSection('overview-basic', '基本信息', content, content.replace(/<[^>]+>/g,''));
    },

    bibleEditable(bible) {
        if (!bible) return '<p>世界观尚未构建</p>';
        const w = bible.world||{}, s = bible.style_contract||{};
        const themes = bible.themes||[], factions = bible.factions||[], conflicts = bible.core_conflicts||[];
        let html = '';

        // World
        let worldHtml = '';
        if (w.geography) worldHtml += `<p><strong>地理：</strong>${this._esc(w.geography)}</p>`;
        if (w.history) worldHtml += `<p><strong>历史：</strong>${this._esc(w.history)}</p>`;
        if (w.culture) worldHtml += `<p><strong>文化：</strong>${this._esc(w.culture)}</p>`;
        if (w.technology_level) worldHtml += `<p><strong>科技/魔法水平：</strong>${this._esc(w.technology_level)}</p>`;
        if (w.magic_system) worldHtml += `<div class="highlight-box"><strong>力量体系：</strong><br>${this._cleanJson(w.magic_system)}</div>`;
        if (w.power_progression) worldHtml += `<p><strong>进阶路径：</strong>${this._cleanJson(w.power_progression)}</p>`;
        if (w.special_rules?.length) worldHtml += `<p><strong>特殊规则：</strong>${w.special_rules.map(r=>this._esc(r)).join('；')}</p>`;
        html += this._editableSection('bible-world', `世界观：${this._esc(w.name||'未命名')}（${w.world_type||''}）`, worldHtml, worldHtml.replace(/<[^>]+>/g,''));
        if (factions.length) html += this._editableSection('bible-factions', `势力（${factions.length}）`, factions.map(f=>`<div class="char-card"><span class="char-name">${this._esc(f.name)}</span><span class="char-role">${this._esc(f.faction_type||'组织')}</span>${f.goal?`<p>${this._esc(f.goal)}</p>`:''}</div>`).join(''), JSON.stringify(factions,null,2));
        html += this._editableSection('bible-style', '文风设定', `<p>语调：${this._L(this.TONE, s.tone)||'未指定'} / 句式：${this._L(this.SENTENCE, s.sentence_style)} / 节奏：${this._L(this.PACING, s.pacing_preference)} / 叙事：${this._L(this.NARRATIVE, s.narrative_distance)}</p>`, JSON.stringify(s,null,2));
        if (themes.length) html += this._editableSection('bible-themes', `主题（${themes.length}）`, themes.map(t=>`<p><strong>${this._esc(t.name)}：</strong>${this._esc(t.description||t.manifestation||'')}</p>`).join(''), JSON.stringify(themes,null,2));
        if (conflicts.length) html += this._editableSection('bible-conflicts', `核心冲突（${conflicts.length}）`, conflicts.map(c=>`<p><strong>${this._L(this.CONFLICT, c.conflict_type)}：</strong>${this._esc(c.description||'')}${c.stakes?`（赌注：${this._esc(c.stakes)}）`:''}</p>`).join(''), JSON.stringify(conflicts,null,2));
        return html;
    },

    charactersEditable(chars) {
        if (!chars?.characters) return '<p>角色尚未创建</p>';
        const list = Object.values(chars.characters);
        if (!list.length) return '<p>角色尚未创建</p>';
        return list.map((c,i) => {
            let cardHtml = '';
            if (c.archetype&&c.archetype!=='None') cardHtml += `<p>原型：${this._L(this.ARCHETYPE, c.archetype)}</p>`;
            cardHtml += `<p>年龄：${c.age||'未知'}岁${c.gender?` / ${this._esc(c.gender)}`:''}</p>`;
            if (c.personality) cardHtml += `<p>性格：${this._esc(c.personality)}</p>`;
            if (c.motivation) cardHtml += `<p>动机：${this._esc(c.motivation)}</p>`;
            if (c.flaw) cardHtml += `<p>缺陷：${this._esc(c.flaw)}</p>`;
            if (c.backstory) cardHtml += `<p>背景：${this._esc(String(c.backstory).substring(0,300))}</p>`;
            if (c.abilities?.length) cardHtml += `<p>能力：${c.abilities.map(a=>this._esc(String(a))).join('、')}</p>`;
            return this._editableSection(`char-${i}`, c.name, cardHtml, JSON.stringify(c,null,2));
        }).join('');
    },

    outlineEditable(outline) {
        if (!outline) return '<p>大纲尚未创建</p>';
        const vols = outline.volumes||[], tps = outline.major_turning_points||[];
        const main = outline.main_plot||[], subs = outline.subplots||[];
        let html = '';
        html += this._editableSection('outline-info', '基本信息', `<p>${this._esc(outline.logline||'')}</p><p>计划章节：${outline.chapter_count||'待定'}章</p>`, JSON.stringify({logline:outline.logline,chapter_count:outline.chapter_count},null,2));
        if (main.length) html += this._editableSection('outline-main', '主线', main.map(a=>`<p><strong>${this._esc(a.name)}</strong>${a.description?`：${this._esc(a.description)}`:''}</p>`).join(''), JSON.stringify(main,null,2));
        if (subs.length) html += this._editableSection('outline-subs', '支线', subs.map(a=>`<p><strong>${this._esc(a.name)}</strong>${a.description?`：${this._esc(a.description)}`:''}</p>`).join(''), JSON.stringify(subs,null,2));
        if (vols.length) html += this._editableSection('outline-vols', `分卷（${vols.length}）`, vols.map((v,i)=>{const n=v.number||i+1;return`<p><strong>卷${n}：${this._esc(v.title)}</strong>（第${v.start_chapter||'?'}-${v.end_chapter||'?'}章）${v.logline?` — ${this._esc(v.logline)}`:''}</p>`;}).join(''), JSON.stringify(vols,null,2));
        if (tps.length) html += this._editableSection('outline-tps', '关键转折点', tps.map(tp=>`<p><strong>${this._L(this.TURNING, tp.turning_type)}（第${tp.chapter||'?'}章）：</strong>${this._esc(tp.description||'')}</p>`).join(''), JSON.stringify(tps,null,2));
        return html;
    },

    projectOverview(p) {
        if (!p?.config) return '<p>加载中...</p>';
        const c = p.config;
        return `<div class="section"><h4>基本信息</h4>
            <p><strong>书名：</strong>${this._esc(c.title||'未设置')}</p>
            <p><strong>灵感：</strong>${this._esc(c.inspiration||'')}</p>
            <p><strong>题材：</strong>${(c.genre||[]).join('、')}</p>
            <p><strong>目标读者：</strong>${this._esc(c.target_readers||'')}</p>
            <p><strong>语调：</strong>${this.TONE[c.tone]||c.tone||'未指定'}</p>
            <p><strong>篇幅：</strong>${c.target_length||'未指定'}（约${(c.target_word_count||0).toLocaleString()}字）</p>
            ${c.style_reference ? `<p><strong>风格参考：</strong>${this._esc(c.style_reference)}</p>` : ''}
        </div>`;
    },

    bibleView(bible) {
        if (!bible) return '<p>世界观尚未构建</p>';
        const w = bible.world||{}, s = bible.style_contract||{};
        const themes = bible.themes||[], factions = bible.factions||[], conflicts = bible.core_conflicts||[];
        return `<div class="section"><h4>世界观：${this._esc(w.name||'未命名')}（${w.world_type||''}）</h4>
            ${w.geography?`<p><strong>地理：</strong>${this._esc(w.geography)}</p>`:''}
            ${w.history?`<p><strong>历史：</strong>${this._esc(w.history)}</p>`:''}
            ${w.culture?`<p><strong>文化：</strong>${this._esc(w.culture)}</p>`:''}
            ${w.technology_level?`<p><strong>科技/魔法水平：</strong>${this._esc(w.technology_level)}</p>`:''}
            ${w.magic_system?`<div class="highlight-box"><strong>力量体系：</strong><br>${this._cleanJson(w.magic_system)}</div>`:''}
            ${w.power_progression?`<p><strong>进阶路径：</strong>${this._cleanJson(w.power_progression)}</p>`:''}
            ${w.special_rules?.length?`<p><strong>特殊规则：</strong>${w.special_rules.map(r=>this._esc(r)).join('；')}</p>`:''}
        </div>
        ${factions.length?`<div class="section"><h4>势力（${factions.length}）</h4>${factions.map(f=>`<div class="char-card"><span class="char-name">${this._esc(f.name)}</span><span class="char-role">${this._esc(f.faction_type||'组织')}</span>${f.goal?`<p>${this._esc(f.goal)}</p>`:''}${f.ideology?`<p>理念：${this._esc(f.ideology)}</p>`:''}</div>`).join('')}</div>`:''}
        <div class="section"><h4>文风设定</h4><p>语调：${this._L(this.TONE, s.tone)||'未指定'} / 句式：${this._L(this.SENTENCE, s.sentence_style)} / 节奏：${this._L(this.PACING, s.pacing_preference)} / 叙事：${this._L(this.NARRATIVE, s.narrative_distance)}</p>${s.preferred_techniques?.length?`<p>技法：${s.preferred_techniques.join('、')}</p>`:''}</div>
        ${themes.length?`<div class="section"><h4>主题（${themes.length}）</h4>${themes.map(t=>`<div class="char-card"><span class="char-name">${this._esc(t.name)}</span><p>${this._esc(t.description||t.manifestation||'')}</p></div>`).join('')}</div>`:''}
        ${conflicts.length?`<div class="section"><h4>核心冲突（${conflicts.length}）</h4>${conflicts.map(c=>`<div class="char-card"><span class="char-name">${this._L(this.CONFLICT, c.conflict_type)}</span><p>${this._esc(c.description||'')}</p>${c.stakes?`<p>赌注：${this._esc(c.stakes)}</p>`:''}</div>`).join('')}</div>`:''}`;
    },

    charactersView(chars) {
        if (!chars?.characters) return '<p>角色尚未创建</p>';
        const list = Object.values(chars.characters);
        if (!list.length) return '<p>角色尚未创建</p>';
        return list.map(c => `<div class="char-card">
            <h4>${this._esc(c.name)}</h4>
            ${c.archetype&&c.archetype!=='None'?`<p>原型：${this._L(this.ARCHETYPE, c.archetype)}</p>`:''}
            <p>年龄：${c.age||'未知'}岁${c.gender?` / ${this._esc(c.gender)}`:''}</p>
            ${c.personality?`<p>性格：${this._esc(c.personality)}</p>`:''}
            ${c.motivation?`<p>动机：${this._esc(c.motivation)}</p>`:''}
            ${c.flaw?`<p>缺陷：${this._esc(c.flaw)}</p>`:''}
            ${c.backstory?`<p>背景：${this._esc(String(c.backstory).substring(0,300))}</p>`:''}
            ${c.abilities?.length?`<p>能力：${c.abilities.map(a=>this._esc(String(a))).join('、')}</p>`:''}
        </div>`).join('');
    },

    outlineView(outline) {
        if (!outline) return '<p>大纲尚未创建</p>';
        const vols = outline.volumes||[], tps = outline.major_turning_points||[];
        const main = outline.main_plot||[], subs = outline.subplots||[];
        return `<div class="section"><h4>${this._esc(outline.title||'未命名')}</h4>${outline.logline?`<p>${this._esc(outline.logline)}</p>`:''}<p>计划章节：${outline.chapter_count||'待定'}章</p></div>
        ${main.length?`<div class="section"><h4>主线</h4>${main.map(a=>`<div class="char-card"><strong>${this._esc(a.name)}</strong>${a.description?`：${this._esc(a.description)}`:''}</div>`).join('')}</div>`:''}
        ${subs.length?`<div class="section"><h4>支线</h4>${subs.map(a=>`<div class="char-card"><strong>${this._esc(a.name)}</strong>${a.description?`：${this._esc(a.description)}`:''}</div>`).join('')}</div>`:''}
        ${vols.length?`<div class="section"><h4>分卷（${vols.length}）</h4>${vols.map((v,i)=>{const n=v.number||i+1;return`<div class="char-card"><span class="char-name">卷${n}：${this._esc(v.title)}</span><p>第${v.start_chapter||'?'}-${v.end_chapter||'?'}章</p>${v.logline?`<p>${this._esc(v.logline)}</p>`:''}${v.emotional_arc?`<p>情感弧线：${this._esc(v.emotional_arc)}</p>`:''}</div>`;}).join('')}</div>`:''}
        ${tps.length?`<div class="section"><h4>关键转折点</h4>${tps.map(tp=>`<div class="char-card"><span class="char-name">${this._L(this.TURNING, tp.turning_type)}</span><span class="char-role">第${tp.chapter||'?'}章</span><p>${this._esc(tp.description||'')}</p></div>`).join('')}</div>`:''}`;
    },

    _CN_NUM: ['','一','二','三','四','五','六','七','八','九','十',
              '十一','十二','十三','十四','十五','十六','十七','十八','十九','二十'],

    _STATUS: { planned:'已规划', writing:'写作中', reviewing:'审核中', polishing:'润色中', done:'已完成', initialized:'未开始' },

    chaptersView(chapters, opts) {
        const hasChapters = chapters && chapters.length;
        const { workflowDone, totalChapters } = opts || {};
        const allDone = workflowDone;
        const partial = hasChapters && !allDone;
        let html = '';
        if (!hasChapters) {
            html += '<p>暂无章节</p>';
            html += `<div class="section"><button class="btn btn-primary btn-lg" onclick="App.startWorkflow()">开始生成</button></div>`;
        } else {
            html += chapters.map(ch => {
                const num = ch.chapter_number || 0;
                const cn = this._CN_NUM[num] || num;
                const st = this._STATUS[ch.status] || ch.status || '未开始';
                return `<div class="card" style="cursor:pointer">
                    <div onclick="App.viewChapter(${num})" style="flex:1">
                        <h3>第${cn}章 ${this._esc(ch.title)}</h3>
                        <span class="status status-${ch.status||'initialized'}">${st}</span>
                    </div>
                    <button class="btn btn-error btn-sm" onclick="event.stopPropagation();App.deleteChapter(${num})" style="margin-left:8px;flex-shrink:0">删除</button>
                </div>`;
            }).join('');
            if (partial) {
                const done = chapters.length;
                html += `<div class="section" style="margin-top:12px"><p>已完成 ${this._CN_NUM[done]||done} 章${totalChapters ? ' / 共 '+totalChapters+' 章' : ''}</p><button class="btn btn-primary" onclick="App.startWorkflow()">继续生成</button></div>`;
            }
        }
        if (allDone) {
            html += `<div class="section" style="margin-top:16px"><h4>导出为</h4><div style="display:flex;gap:8px"><button class="btn btn-secondary btn-sm" onclick="App.exportDocx()">DOCX</button><button class="btn btn-secondary btn-sm" onclick="App.exportMarkdown()">Markdown</button></div></div>`;
        }
        return html;
    },

    reviewScores(scores) {
        if (!scores) return '';
        const labels = { consistency:'一致性', character:'角色', pacing:'节奏', hook:'钩子', style:'文风', ai_flavor:'自然度', reader_engagement:'沉浸感', continuation_likelihood:'追读意愿' };
        return `<div class="score-grid">${Object.entries(scores).map(([k,v])=>`<div class="score-item"><div class="score-value">${Number(v).toFixed(1)}</div><div class="score-label">${labels[k]||k}</div></div>`).join('')}</div>`;
    },

    issuesList(issues) {
        if (!issues?.length) return '<p>无问题</p>';
        return issues.map(i => `<div class="issue ${i.severity||''}"><span class="severity-tag ${i.severity||''}">${this._L(this.SEVERITY, i.severity)||'提示'}</span>${i.description?this._esc(i.description):''}${i.suggestion?`<br><em>建议：${this._esc(i.suggestion)}</em>`:''}</div>`).join('');
    },

    phaseConfirmPanel(phase) {
        const labels = { platform_scan:'平台扫榜', topic_selection:'选题研究', mini_arc_outline:'小事件大纲', bible_construction:'世界观构建', character_creation:'角色创建', master_outline:'大纲生成', chapter_planning:'章节规划', chapter_writing:'章节写作' };
        const hints = { platform_scan:'扫榜结果是否符合预期？确认后进入选题研究。', topic_selection:'选题方向是否满意？确认后生成小事件大纲。', mini_arc_outline:'大纲是否符合预期？确认后进入世界观构建。', bible_construction:'对世界观有什么想法？确认后进入角色创建。', character_creation:'对角色有什么想法？确认后进入大纲生成。', master_outline:'对大纲有什么想法？确认后开始章节生成。' };
        const label = labels[phase]||phase;
        return `<div class="confirm-panel phase-confirm" data-phase="${phase}"><h3>${label} 完成</h3><p>${hints[phase]||'确认后继续下一阶段。'}</p>
            <div class="form-group"><label>补充想法（可选）</label><textarea id="inspiration-input" placeholder="把你的想法写在这里..." style="width:100%;padding:10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);min-height:60px;font-family:inherit;font-size:0.9rem;"></textarea></div>
            <div class="confirm-actions"><button class="btn btn-success" onclick="App.confirmPhase()">确认，继续</button></div></div>`;
    },

    confirmPanel(chapterNumber, review) {
        return `<div class="confirm-panel"><h3>第${chapterNumber}章 审阅完成</h3>
            ${review?`<p>总分: <strong>${Number(review.overall_score).toFixed(1)}/10</strong></p>${this.reviewScores(review.dimension_scores)}<h4 style="margin-top:16px">问题 (${review.issues?.length||0})</h4>${this.issuesList(review.issues)}`:''}
            <div class="confirm-actions"><button class="btn btn-success" onclick="App.confirmDecision('accept')">接受</button><button class="btn btn-warning" onclick="App.confirmDecision('revise')">修改</button><button class="btn btn-error" onclick="App.confirmDecision('rewrite')">重写</button></div>
            <div style="margin-top:12px"><textarea id="feedback-input" placeholder="修改意见..." style="width:100%;padding:10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);min-height:60px"></textarea></div></div>`;
    },

    topicResearchView() {
        return `<div class="section">
            <h4>市场调研 · 平台扫榜</h4>
            <p style="color:var(--text-secondary);margin-bottom:16px">
                粘贴飞卢和/或番茄小说榜单页面的 HTML 内容（支持纯文本，最多 15000 字），
                AI 将自动提取热门题材、分析趋势并生成选题建议。
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div class="form-group">
                    <label for="scan-feilu"><strong>飞卢小说榜单</strong></label>
                    <textarea id="scan-feilu" placeholder="粘贴飞卢榜单页面内容..." style="width:100%;min-height:200px;padding:10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);font-family:monospace;font-size:0.85rem;resize:vertical"></textarea>
                </div>
                <div class="form-group">
                    <label for="scan-fanqie"><strong>番茄小说榜单</strong></label>
                    <textarea id="scan-fanqie" placeholder="粘贴番茄榜单页面内容..." style="width:100%;min-height:200px;padding:10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);font-family:monospace;font-size:0.85rem;resize:vertical"></textarea>
                </div>
            </div>
            <div class="confirm-actions" style="margin-top:16px">
                <button class="btn btn-primary" onclick="App.submitScan()">开始扫榜分析</button>
            </div>
            <p class="hint" style="margin-top:8px;color:var(--text-secondary);font-size:0.8rem">
                提示：可以使用浏览器开发者工具 (F12) 复制榜单页面的 HTML，或在 Playwright 中抓取页面内容后粘贴。
                扫榜完成后将自动进入「选题研究 → 小事件大纲」流程，之后再进入世界观构建。
            </p>
        </div>`;
    },

    _esc(s) {
        if (!s) return '';
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    },
};
