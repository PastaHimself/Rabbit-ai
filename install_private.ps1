param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"

function Resolve-PythonInvocation {
    param([string]$Preferred)

    $preferredCommand = Get-Command $Preferred -ErrorAction SilentlyContinue
    if ($preferredCommand) {
        return @{
            Executable = $Preferred
            PrefixArgs = @()
            Display = $Preferred
        }
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        return @{
            Executable = "py"
            PrefixArgs = @("-3")
            Display = "py -3"
        }
    }

    throw "Python was not found on PATH. Install Python 3.11+ or pass -PythonCommand with the full executable name."
}

$python = Resolve-PythonInvocation -Preferred $PythonCommand

Write-Host "Installing Rabbit AI from the local checkout..."
& $python.Executable @($python.PrefixArgs + @("-m", "pip", "install", "-r", "requirements.txt"))
& $python.Executable @($python.PrefixArgs + @("-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."))

Write-Host "Install complete. Run 'rabbit-ai' or '$($python.Display) -m rabbit_ai'."
