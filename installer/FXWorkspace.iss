; FX Workspace - Windows setup wizard (Inno Setup 6). Build it with installer\build.ps1, which stages the files
; first: only what git publishes (no chats, keys, .venv or models) plus the packaged desktop app (desktop\dist),
; and passes the version from the VERSION file.
;
; The wizard installs per user (no admin), into %LOCALAPPDATA%\Programs\FX Workspace by default - the app writes
; its chats and settings next to itself (data\), so it can't live in Program Files. Then it runs install.ps1
; unattended: Python, the app's packages, llama.cpp for this GPU and, if chosen, the starter model and the image
; models. Everything it downloads comes from the official source and is checked by SHA-256 (install.ps1).
;
; UPDATING: run a newer installer over an installed copy. It finds it (same AppId), asks nothing it already knows
; (folder, models folder, components, shortcuts), closes the app, replaces the app's own files, removes the ones
; the new version no longer has (.fx-manifest: the list of files the last install put there - nothing else is ever
; deleted), and runs install.ps1 -Update, which brings the Python packages and the app's own llama.cpp up to date
; and removes what the old version left behind: packages it no longer uses, the old engine build's files,
; __pycache__, old models.ini backups (update_cleanup.py). pip keeps no download cache.
; Kept as they are: data\ (chats, pictures, videos, music, memory), settings.json, keys.local.bat, models.ini, your
; models, and any file that isn't the app's.
;
; install.ps1 runs hidden: its output shows line by line on the wizard's progress page and goes to
; data\logs\setup.log. Settings > About > Update now runs this wizard with /SILENT /FXRESTART=1: no questions,
; a progress window, and the app opens again when it's done (app_version.py).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "FX Workspace"
#define AppPublisher "Fuxion"
#define AppExe "desktop-app\FX Workspace.exe"
#define Stage "stage"
#define UninstKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\{6E0F3B52-8F7A-4C1E-9B2D-FA11C0DE2026}_is1"

[Setup]
AppId={{6E0F3B52-8F7A-4C1E-9B2D-FA11C0DE2026}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/fuxion64-ctrl/FX-Workspace
AppSupportURL=https://github.com/fuxion64-ctrl/FX-Workspace/issues
AppUpdatesURL=https://github.com/fuxion64-ctrl/FX-Workspace/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
UsePreviousSetupType=yes
UsePreviousTasks=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
LicenseFile={#Stage}\app\LICENSE
SetupIconFile={#Stage}\app\desktop\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
WizardSizePercent=110
Compression=lzma2/max
SolidCompression=yes
OutputDir=output
OutputBaseFilename=FX-Workspace-Setup-{#AppVersion}
ChangesEnvironment=yes
; The app is closed by close-app.ps1 (PrepareToInstall), gracefully, so it can stop llama-server itself.
CloseApplications=no
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "standard"; Description: "Standard - the app, the local model engine and a starter chat model (~3 GB)"
Name: "full"; Description: "Everything - also the image models (~25 GB, NVIDIA GPU)"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "app"; Description: "FX Workspace (the app and its Python environment)"; Types: standard full custom; Flags: fixed
Name: "engine"; Description: "llama.cpp - runs chat models on this PC (picked for your GPU)"; Types: standard full custom
Name: "model"; Description: "Starter chat model - Qwen3 4B (2.4 GB download)"; Types: standard full
Name: "images"; Description: "Image models - FLUX.2, Z-Image and more (~22 GB download, NVIDIA GPU)"; Types: full

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start FX Workspace when I sign in"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
; Read by the code before anything is copied: the new version's file list, and the script that closes the app.
Source: "{#Stage}\app\.fx-manifest"; Flags: dontcopy
Source: "close-app.ps1"; Flags: dontcopy
Source: "{#Stage}\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\data"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Your private, local AI assistant"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "--hidden"; WorkingDir: "{app}"; Tasks: startup

[Registry]
; Models and engines on another drive: the app reads AGENTIC_HOME (paths.py). Only set when it isn't the default.
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "AGENTIC_HOME"; ValueData: "{code:ModelsDir}"; Check: CustomModelsDir; Flags: uninsdeletevalue

[Run]
; install.ps1 runs from CurStepChanged (ssPostInstall), with its output on the progress page; after an Update now
; (silent) the app is opened again from there too - a [Run] entry without postinstall would run BEFORE setup.
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\static"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files; Name: "{app}\.fx-manifest"

[Code]
var
  ModelsPage: TInputDirWizardPage;
  PrevVersion: String;            { the installed version when this is an update, else '' }

{ ---- the installed copy ---------------------------------------------------------------------------------- }
function InstalledVersion: String;
begin
  Result := '';
  if not RegQueryStringValue(HKCU, '{#UninstKey}', 'DisplayVersion', Result) then
    RegQueryStringValue(HKLM, '{#UninstKey}', 'DisplayVersion', Result);
end;

function IsUpdate: Boolean;
begin
  Result := PrevVersion <> '';
end;

{ '1.10.2' vs '1.9.0': -1, 0 or 1 }
function VersionPart(var S: String): Integer;
var P: Integer;
begin
  P := Pos('.', S);
  if P = 0 then begin Result := StrToIntDef(S, 0); S := ''; end
  else begin Result := StrToIntDef(Copy(S, 1, P - 1), 0); S := Copy(S, P + 1, Length(S)); end;
end;

function CompareVersions(A, B: String): Integer;
var I, X, Y: Integer;
begin
  Result := 0;
  for I := 1 to 3 do
  begin
    X := VersionPart(A); Y := VersionPart(B);
    if X > Y then begin Result := 1; Exit; end;
    if X < Y then begin Result := -1; Exit; end;
  end;
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  PrevVersion := InstalledVersion;
  if IsUpdate and (CompareVersions(PrevVersion, '{#AppVersion}') > 0) then
    Result := SuppressibleMsgBox('FX Workspace ' + PrevVersion + ' is installed - newer than this installer (' + '{#AppVersion}' + ').' + #13#10#13#10 +
                     'Install the older version over it anyway? Your chats and settings are kept either way.',
                     mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES;
end;

{ ---- pages ----------------------------------------------------------------------------------------------- }
procedure InitializeWizard;
begin
  ModelsPage := CreateInputDirPage(wpSelectComponents,
    'Where should models go?',
    'Chat, image and video models are large (2 GB to 20+ GB each).',
    'FX Workspace keeps its model engine and the models you download in this folder. Pick a drive with plenty of ' +
    'free space - an SSD loads models fastest. You can add models from other folders later in Settings > Models.',
    False, '');
  ModelsPage.Add('Models and engines folder:');
  ModelsPage.Values[0] := GetPreviousData('ModelsDir', '');
  if IsUpdate then
  begin
    WizardForm.Caption := 'Update FX Workspace';
    WizardForm.ReadyLabel.Caption := 'Setup is ready to update FX Workspace. Click Install to continue.';
  end;
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'ModelsDir', ModelsPage.Values[0]);
end;

{ An update asks nothing it already knows: the same folder, models folder, components and shortcuts. }
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := IsUpdate and ((PageID = wpLicense) or (PageID = wpSelectDir) or (PageID = wpSelectComponents) or
                          (PageID = wpSelectTasks) or (PageID = ModelsPage.ID));
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = ModelsPage.ID) and (ModelsPage.Values[0] = '') then
    ModelsPage.Values[0] := AddBackslash(WizardDirValue) + 'local';
end;

function ModelsDir(Param: String): String;
begin
  if Trim(ModelsPage.Values[0]) = '' then
    ModelsPage.Values[0] := AddBackslash(WizardDirValue) + 'local';
  Result := RemoveBackslashUnlessRoot(ModelsPage.Values[0]);
end;

function CustomModelsDir: Boolean;
begin
  Result := CompareText(ModelsDir(''), AddBackslash(ExpandConstant('{app}')) + 'local') <> 0;
end;

function SetupFlags(Param: String): String;
begin
  Result := ' -Quiet';
  if IsUpdate then Result := Result + ' -Update';
  if not WizardIsComponentSelected('engine') then Result := Result + ' -SkipLlama';
  if not WizardIsComponentSelected('model') then Result := Result + ' -SkipModel';
  if WizardIsComponentSelected('images') then Result := Result + ' -Images';
end;

function ReopenAfterUpdate: Boolean;
begin
  Result := ExpandConstant('{param:FXRESTART|0}') = '1';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = ModelsPage.ID) and (Trim(ModelsPage.Values[0]) = '') then
  begin
    MsgBox('Pick a folder for the models.', mbError, MB_OK);
    Result := False;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo,
  MemoGroupInfo, MemoTasksInfo: String): String;
begin
  if IsUpdate then
    Result := 'Update FX Workspace ' + PrevVersion + '  ->  ' + '{#AppVersion}' + NewLine + NewLine +
              MemoDirInfo + NewLine + NewLine +
              'Kept exactly as they are:' + NewLine +
              Space + 'your chats, pictures, videos, music, memory and settings (data\)' + NewLine +
              Space + 'your API keys (keys.local.bat) and model list (models.ini)' + NewLine +
              Space + 'your models (' + ModelsDir('') + ')' + NewLine + NewLine +
              'Brought up to date, with the old versions removed:' + NewLine +
              Space + 'the app''s Python packages (ones it no longer uses are uninstalled)' + NewLine +
              Space + 'the model engine (llama.cpp), when it''s the one setup installed' + NewLine + NewLine +
              'FX Workspace is closed first if it is running.'
  else
    Result := MemoDirInfo + NewLine + NewLine +
              'Models and engines folder:' + NewLine + Space + ModelsDir('') + NewLine + NewLine +
              MemoComponentsInfo + NewLine + NewLine + MemoTasksInfo + NewLine + NewLine +
              'After copying the files, setup downloads Python (if missing), the app''s packages and what you chose. ' +
              'Every download is checked against its published SHA-256.';
end;

{ ---- installing ------------------------------------------------------------------------------------------ }
{ Close the running app (gracefully first, so it stops llama-server itself) before any file is replaced. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var Code: Integer;
begin
  Result := '';
  if not DirExists(ExpandConstant('{app}')) then Exit;
  ExtractTemporaryFile('close-app.ps1');
  Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\close-app.ps1') +
       '" -App "' + ExpandConstant('{app}') + '"', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

{ Remove the files the last install put here that the new version no longer has. Only files in the old
  .fx-manifest - the installer's own list - are ever considered, so nothing of yours is touched. }
procedure RemoveStaleFiles;
var OldList, NewList: TStringList; I: Integer; F, Low: String;
begin
  if not FileExists(ExpandConstant('{app}\.fx-manifest')) then Exit;
  ExtractTemporaryFile('.fx-manifest');
  OldList := TStringList.Create;
  NewList := TStringList.Create;
  try
    OldList.LoadFromFile(ExpandConstant('{app}\.fx-manifest'));
    NewList.LoadFromFile(ExpandConstant('{tmp}\.fx-manifest'));
    NewList.CaseSensitive := False;
    NewList.Sorted := True;
    for I := 0 to OldList.Count - 1 do
    begin
      F := Trim(OldList[I]);
      Low := Lowercase(F);
      if (F = '') or (Pos('..', F) > 0) or (Pos(':', F) > 0) or (Copy(F, 1, 1) = '\') then Continue;
      if (Copy(Low, 1, 5) = 'data\') or (Copy(Low, 1, 6) = 'local\') or (Copy(Low, 1, 6) = '.venv\') then Continue;
      if NewList.IndexOf(F) < 0 then
        DeleteFile(ExpandConstant('{app}\') + F);
    end;
  finally
    OldList.Free;
    NewList.Free;
  end;
end;

{ ---- the setup step (install.ps1), hidden, its output on the progress page -------------------------------- }
var
  SetupTail: TStringList;          { the last lines, for the message when it fails }
  SetupLogFile: String;            { data\logs\setup.log }

procedure SetupOutput(const S: String; const Error, FirstLine: Boolean);
var Line: String;
begin
  Line := Trim(S);
  if Line = '' then Exit;
  SaveStringToFile(SetupLogFile, Line + #13#10, True);
  SetupTail.Add(Line);
  if SetupTail.Count > 8 then SetupTail.Delete(0);
  if Copy(Line, 1, 3) = '== ' then
    WizardForm.StatusLabel.Caption := Copy(Line, 4, 120) + '...'
  else
    WizardForm.FilenameLabel.Caption := Copy(Line, 1, 140);
end;

procedure RunSetupScript;
var Code: Integer; Ok: Boolean; Params: String;
begin
  SetupLogFile := ExpandConstant('{app}\data\logs\setup.log');
  ForceDirectories(ExtractFilePath(SetupLogFile));
  SaveStringToFile(SetupLogFile, 'FX Workspace {#AppVersion} setup - ' + GetDateTimeString('yyyy-mm-dd hh:nn', '-', ':') + #13#10, False);
  { No redirection inside PowerShell: under 'Stop', Windows PowerShell 5.1 turns a redirected native program's
    stderr line (a pip notice) into a terminating error. The wizard reads stdout and stderr itself. }
  Params := '-NoProfile -ExecutionPolicy Bypass -Command "$env:AGENTIC_HOME=''' + ModelsDir('') + '''; ' +
            '& ''' + ExpandConstant('{app}\install.ps1') + ''' -Yes -SkipShortcut' + SetupFlags('') + '; exit $LASTEXITCODE"';
  SetupTail := TStringList.Create;
  try
    if IsUpdate then
      WizardForm.StatusLabel.Caption := 'Updating the app''s packages and the model engine...'
    else
      WizardForm.StatusLabel.Caption := 'Setting up Python, the app''s packages and the model engine...';
    WizardForm.ProgressGauge.Style := npbstMarquee;
    Ok := ExecAndLogOutput('powershell.exe', Params, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code, @SetupOutput);
    WizardForm.ProgressGauge.Style := npbstNormal;
    if (not Ok) or (Code <> 0) then
      SuppressibleMsgBox('Setup copied the app, but its last step hit a problem:' + #13#10#13#10 + SetupTail.Text + #13#10 +
                         'The full log is in ' + SetupLogFile + '. Start FX Workspace when you''re online again: it finishes ' +
                         'setting itself up (1.4.8) - no terminal needed.',
                         mbError, MB_OK, IDOK);
  finally
    SetupTail.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Code: Integer;
begin
  if (CurStep = ssInstall) and IsUpdate then
  begin
    { From 1.4.1 and older: remember the old requirements, so packages only they needed can go (update_cleanup.py). }
    if FileExists(ExpandConstant('{app}\requirements.txt')) and DirExists(ExpandConstant('{app}\.venv')) and
       not FileExists(ExpandConstant('{app}\.venv\fx-requirements.txt')) then
      FileCopy(ExpandConstant('{app}\requirements.txt'), ExpandConstant('{app}\.venv\fx-requirements.txt'), False);
    RemoveStaleFiles;
  end;
  if CurStep = ssPostInstall then
  begin
    RunSetupScript;
    if ReopenAfterUpdate then        { Update now: the app comes back once everything is done }
      ExecAsOriginalUser(ExpandConstant('{app}\{#AppExe}'), '', ExpandConstant('{app}'), SW_SHOWNORMAL, ewNoWait, Code);
  end;
end;

{ ---- uninstalling ---------------------------------------------------------------------------------------- }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if DirExists(ExpandConstant('{app}\data')) then
      if (not UninstallSilent) and (SuppressibleMsgBox('Also delete your chats, pictures, videos, music and settings?' + #13#10 + #13#10 +
                'Choose No to keep them (in ' + ExpandConstant('{app}\data') + ') for a later reinstall.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES) then
      begin
        DelTree(ExpandConstant('{app}\data'), True, True, True);
        DeleteFile(ExpandConstant('{app}\keys.local.bat'));       { your API keys }
        DeleteFile(ExpandConstant('{app}\models.ini'));
        DeleteFile(ExpandConstant('{app}\settings.json'));         { your name, theme and preferences }
      end;
    if DirExists(ExpandConstant('{app}\local')) then
      if (not UninstallSilent) and (SuppressibleMsgBox('Also delete the downloaded models and the model engine?' + #13#10 + #13#10 +
                ExpandConstant('{app}\local') + #13#10 + 'Choose No to keep them - they can be several GB to re-download.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES) then
        DelTree(ExpandConstant('{app}\local'), True, True, True);
    RemoveDir(ExpandConstant('{app}'));                            { only when nothing is left in it }
  end;
end;
