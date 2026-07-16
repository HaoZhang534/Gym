"""Lazy exports for OSWorld getter functions."""

from importlib import import_module


_GETTER_MODULES = {
    "get_default_search_engine": ".chrome",
    "get_cookie_data": ".chrome",
    "get_bookmarks": ".chrome",
    "get_open_tabs_info": ".chrome",
    "get_pdf_from_url": ".chrome",
    "get_shortcuts_on_desktop": ".chrome",
    "get_history": ".chrome",
    "get_page_info": ".chrome",
    "get_enabled_experiments": ".chrome",
    "get_chrome_language": ".chrome",
    "get_chrome_font_size": ".chrome",
    "get_chrome_color_scheme": ".chrome",
    "get_chrome_appearance_mode_ui": ".chrome",
    "get_profile_name": ".chrome",
    "get_number_of_search_results": ".chrome",
    "get_googledrive_file": ".chrome",
    "get_active_tab_info": ".chrome",
    "get_enable_do_not_track": ".chrome",
    "get_enable_enhanced_safety_browsing": ".chrome",
    "get_enable_safe_browsing": ".chrome",
    "get_new_startup_page": ".chrome",
    "get_find_unpacked_extension_path": ".chrome",
    "get_data_delete_automacally": ".chrome",
    "get_active_tab_html_parse": ".chrome",
    "get_active_tab_url_parse": ".chrome",
    "get_gotoRecreationPage_and_get_html_content": ".chrome",
    "get_url_dashPart": ".chrome",
    "get_active_url_from_accessTree": ".chrome",
    "get_find_installed_extension_name": ".chrome",
    "get_info_from_website": ".chrome",
    "get_macys_product_url_parse": ".chrome",
    "get_url_path_parse": ".chrome",
    "get_cloud_file": ".file",
    "get_vm_file": ".file",
    "get_cache_file": ".file",
    "get_content_from_vm_file": ".file",
    "get_vm_command_line": ".general",
    "get_vm_terminal_output": ".general",
    "get_vm_command_error": ".general",
    "get_gimp_config_file": ".gimp",
    "get_audio_in_slide": ".impress",
    "get_background_image_in_slide": ".impress",
    "get_vm_screen_size": ".info",
    "get_vm_window_size": ".info",
    "get_vm_wallpaper": ".info",
    "get_list_directory": ".info",
    "get_rule": ".misc",
    "get_accessibility_tree": ".misc",
    "get_rule_relativeTime": ".misc",
    "get_time_diff_range": ".misc",
    "get_replay": ".replay",
    "get_vlc_playing_info": ".vlc",
    "get_vlc_config": ".vlc",
    "get_default_video_player": ".vlc",
    "get_vscode_config": ".vscode",
    "get_conference_city_in_order": ".calc",
}

__all__ = sorted(_GETTER_MODULES)


def __getattr__(name):
    module_name = _GETTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
