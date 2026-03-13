#!/usr/bin/env node

const path = require('path');

const MINIMUM_PYTHON_VERSION = '3.10';
const MINIMUM_PYTHON_MAJOR = 3;
const MINIMUM_PYTHON_MINOR = 10;

function parsePythonVersion(versionOutput) {
  if (!versionOutput) {
    return null;
  }

  const match = String(versionOutput).match(/Python\s+(\d+)\.(\d+)\.(\d+)/i);
  if (!match) {
    return null;
  }

  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
  };
}

function isSupportedPythonVersion(versionOutput) {
  const parsed = parsePythonVersion(versionOutput);
  if (!parsed) {
    return false;
  }

  if (parsed.major !== MINIMUM_PYTHON_MAJOR) {
    return parsed.major > MINIMUM_PYTHON_MAJOR;
  }

  return parsed.minor >= MINIMUM_PYTHON_MINOR;
}

function getUnsupportedPythonMessage(versionOutput) {
  const detected = versionOutput || 'unknown version';
  return `FastMCP requires Python ${MINIMUM_PYTHON_VERSION}+ (detected: ${detected}).`;
}

function getMissingPythonMessage() {
  return `Please install Python ${MINIMUM_PYTHON_VERSION}+ and ensure "python" or "python3" is in your PATH.`;
}

// builds the candidate list, preferring venv python if VIRTUAL_ENV is set
function buildPythonCandidates() {
  const candidates = [];

  if (process.env.VIRTUAL_ENV) {
    const venvBin = path.join(process.env.VIRTUAL_ENV, 'bin');
    candidates.push(path.join(venvBin, 'python'));
    candidates.push(path.join(venvBin, 'python3'));
  }

  candidates.push('python', 'python3');
  return candidates;
}

async function resolveSupportedPythonCommand(runCommand, candidates = buildPythonCandidates()) {
  let unsupportedVersion = null;

  for (const command of candidates) {
    try {
      const version = await runCommand(command, ['--version']);

      if (isSupportedPythonVersion(version)) {
        return {
          status: 'supported',
          command,
          version: version || 'Unknown version',
        };
      }

      unsupportedVersion = version || 'Unknown version';
    } catch {
      // Try next candidate
    }
  }

  if (unsupportedVersion) {
    return {
      status: 'unsupported',
      version: unsupportedVersion,
      message: getUnsupportedPythonMessage(unsupportedVersion),
    };
  }

  return {
    status: 'missing',
    message: getMissingPythonMessage(),
  };
}

module.exports = {
  MINIMUM_PYTHON_VERSION,
  getMissingPythonMessage,
  getUnsupportedPythonMessage,
  isSupportedPythonVersion,
  parsePythonVersion,
  resolveSupportedPythonCommand,
};
