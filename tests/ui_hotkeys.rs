use coding_agent_search::ui::tui::footer_legend;

#[test]
fn footer_mentions_editor_and_clear_keys() {
    let long = footer_legend(true);
    assert!(long.contains("Enter/F8"));
    assert!(long.contains("F11 clear"));
    assert!(long.contains("F7 context"));
}
