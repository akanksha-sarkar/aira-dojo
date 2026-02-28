from dojo.utils.code_parsing import format_code

def set_system_path_code() -> str:
    """Set system path code to include custom directories."""
    system_path_code = f"""
                            import os
                            import sys
                            sys.path.insert(0, "/work")
                            print("✅ Added /work to sys.path")
                            print("Working dir:", os.getcwd())
                        """
    return format_code(system_path_code)