/**
 * AI Agents Settings Page
 */
import $ from "jquery";
import _ from "lodash";

import * as channel from "./channel.ts";
import * as dialog_widget from "./dialog_widget.ts";
import {$t, $t_html} from "./i18n.ts";
import * as loading from "./loading.ts";
import * as overlays from "./overlays.ts";
import {page_params} from "./page_params.ts";
import * as settings_data from "./settings_data.ts";
import * as ui_report from "./ui_report.ts";

export function initialize(): void {
    // Initialize AI agents when settings page loads
    
    // Add AI agents section to settings
    add_ai_agents_section();
}

function add_ai_agents_section(): void {
    const $settings_panel = $("#settings_panel");
    
    // Check if AI agents section already exists
    if ($("#ai-agents-settings").length > 0) {
        return;
    }
    
    const ai_section_html = `
        <div id="ai-agents-settings" class="settings-section" data-name="ai-agents">
            <div class="settings-section-title">
                <h3>${$t({defaultMessage: "AI Agents"})}</h3>
                <div class="alert-notification" id="ai-status-indicator"></div>
            </div>
            
            <div class="settings-section-content">
                <div class="ai-overview">
                    <p>${$t({defaultMessage: "AI agents provide intelligent assistance using local Ollama models."})}</p>
                </div>
                
                <div class="ai-controls">
                    <button class="ai-status-button" id="ai-status-button">
                        <i class="fa fa-info-circle"></i>
                        ${$t({defaultMessage: "Check Status"})}
                    </button>
                    
                    <button class="ai-chat-button" id="ai-chat-button">
                        <i class="fa fa-comments"></i>
                        ${$t({defaultMessage: "AI Chat"})}
                    </button>
                </div>
                
                <div class="ai-info" id="ai-info" style="display: none;">
                    <div class="ai-health-status"></div>
                    <div class="ai-models-summary"></div>
                </div>
            </div>
        </div>
    `;
    
    // Insert AI agents section after existing sections
    $settings_panel.append(ai_section_html);
    
    // Bind event handlers
    $("#ai-status-button").on("click", show_ai_status);
    $("#ai-chat-button").on("click", show_ai_chat);
    
    // Update status indicator
    update_ai_status_indicator();
    
    // Set up periodic status updates
    setInterval(update_ai_status_indicator, 30000); // Update every 30 seconds
}

function update_ai_status_indicator(): void {
    const $indicator = $("#ai-status-indicator");
    const health_status = AIAgentsManager.get_health_status();
    
    if (!health_status) {
        $indicator.html(`
            <span class="ai-status-indicator disabled">
                <i class="fa fa-question-circle"></i>
                ${$t({defaultMessage: "Unknown"})}
            </span>
        `);
        return;
    }
    
    const status_class = health_status.status;
    const status_icon = health_status.status === "healthy" ? "check-circle" : 
                       health_status.status === "unhealthy" ? "times-circle" : 
                       "question-circle";
    const status_text = health_status.status.toUpperCase();
    
    $indicator.html(`
        <span class="ai-status-indicator ${status_class}">
            <i class="fa fa-${status_icon}"></i>
            ${status_text}
        </span>
    `);
}

function show_ai_status(): void {
    ai_agents.show_ai_status_modal();
}

function show_ai_chat(): void {
    ai_agents.show_ai_chat_modal();
}

// Export for use in other modules
export {add_ai_agents_section, update_ai_status_indicator}; 