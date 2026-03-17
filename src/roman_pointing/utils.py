import os


def get_cache_dir():
    """
    Finds the home directory for the system and generates a cache dir as needed

    Returns:
        str:
            Path to cache directory
    """

    # POSIX system
    if os.name == "posix":
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find POSIX home directory")
    # Windows system
    elif os.name == "nt":
        # msys shell
        if "MSYSTEM" in os.environ and os.environ.get("HOME"):
            homedir = os.environ["HOME"]
        # network home
        elif "HOMESHARE" in os.environ:
            homedir = os.environ["HOMESHARE"]
        # local home
        elif "HOMEDRIVE" in os.environ and "HOMEPATH" in os.environ:
            homedir = os.path.join(os.environ["HOMEDRIVE"], os.environ["HOMEPATH"])
        # user profile?
        elif "USERPROFILE" in os.environ:
            homedir = os.path.join(os.environ["USERPROFILE"])
        # something else?
        else:
            try:
                import winreg as wreg

                shell_folders = (
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
                )
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)
                homedir = wreg.QueryValueEx(key, "Personal")[0]
                key.Close()
            except Exception:
                # try home before giving up
                if "HOME" in os.environ:
                    homedir = os.environ["HOME"]
                else:
                    raise OSError("Could not find Windows home directory")
    else:
        # some other platform? try HOME to see if it works
        if "HOME" in os.environ:
            homedir = os.environ["HOME"]
        else:
            raise OSError("Could not find home directory on your platform")

    assert os.path.isdir(homedir) and os.access(homedir, os.R_OK | os.W_OK | os.X_OK), (
        f"Identified {homedir} as home directory, but it does not exist "
        "or is not accessible/writeable"
    )

    path = os.path.join(homedir, ".corgi", "cache")
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except PermissionError:
            print("Cannot create directory: {}".format(path))

    # ensure everything worked out
    assert os.access(path, os.F_OK), "Directory {} does not exist".format(path)
    assert os.access(path, os.R_OK), "Cannot read from directory {}".format(path)
    assert os.access(path, os.W_OK), "Cannot write to directory {}".format(path)
    assert os.access(path, os.X_OK), "Cannot execute directory {}".format(path)

    return path
