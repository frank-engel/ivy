# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

added_files = [

("image_velocimetry_tools\\gui\\icons\\*", "icons"),
("bin", "bin"),
('docs\\_build\\html', 'documentation')

]
a = Analysis(
    ['app.py'],
    pathex=['C:\\REPOS\\ivy'],
    binaries=[],
    datas=added_files,
    hiddenimports=['image_velocimetry_tools.gui.stiv_processor'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

splash = Splash(
    'image_velocimetry_tools\\gui\\icons\\IVy_Logo.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=False,)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    name='IVyTools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='file_version_info.txt',
    icon='image_velocimetry_tools\\gui\\icons\\IVy_Logo.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    splash.binaries,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IVyTools'
)
