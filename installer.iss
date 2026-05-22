; Whisper 语音转文字 - Inno Setup 安装脚本
; 生成 Windows 安装程序 (.exe)

#define MyAppName "Whisper 语音转文字"
#define MyAppNameEn "WhisperSTT"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "dedeyuyu"
#define MyAppURL "https://github.com/dedeyuyu/Speech-to-text"
#define MyAppExeName "WhisperSTT.exe"

[Setup]
; 基本设置
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppNameEn}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=

; 输出设置
OutputDir=installer_output
OutputBaseFilename=WhisperSTT_Setup_{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=120

; 图标
SetupIconFile=image.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; 权限：不需要管理员（用户安装）
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; 最低系统要求
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked
Name: "startmenu"; Description: "创建开始菜单快捷方式"; GroupDescription: "附加任务:"; Flags: checkedonce

[Files]
; 主程序
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; 应用图标
Source: "image.ico"; DestDir: "{app}"; Flags: ignoreversion

; 必读文件
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
; 开始菜单
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\image.ico"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"

; 桌面快捷方式（可选）
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\image.ico"; Tasks: desktopicon

; 系统托盘自启（写入注册表，用户可在设置中控制）

[Registry]
; 不强制自启动，由用户在应用设置中控制

[Run]
; 安装完成后启动应用
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; 卸载时清理注册表自启项
Filename: "reg"; Parameters: "delete ""HKCU\Software\Microsoft\Windows\CurrentVersion\Run"" /v WhisperSTT /f"; Flags: runhidden; StatusMsg: "清理自启动项…"

[Code]
// 安装前检查 .NET/VC++ 依赖（可选提示）
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // 安装后操作（如有需要）
  end;
end;
