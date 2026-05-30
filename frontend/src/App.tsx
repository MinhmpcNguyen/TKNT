import React, { useEffect, useState } from 'react';
import { clearStoredAuthToken, fetchInventory, getCurrentUser, listGeneratedRenders, listSavedLayouts, logoutUser } from './api';
import { AuthScreen } from './components/AuthScreen';
import { AccountPanel } from './components/AccountPanel';
import { Canvas2D } from './components/Canvas2D';
import { LibraryPanel } from './components/LibraryPanel';
import { ProfileModal } from './components/ProfileModal';
import { UserInputPanel } from './components/UserInputPanel';
import { Toolbar } from './components/Toolbar';
import { LayoutVariantsGallery } from './components/LayoutVariantsGallery';
import { ThreeDPreviewPanel } from './components/ThreeDPreviewPanel';
import { isEvalHooksEnabled } from './evalHooks';
import { useEditorStore } from './store';
import sampleFloorplan from './sample_floorplan.json';

export const App: React.FC = () => {
  const {
    authResolved,
    currentUser,
    tenantId,
    setAuthResolved,
    setAuthSession,
    clearAuthSession,
    setFloorplan,
    setBaseFloorplan,
    setInventory,
    setSavedLayouts,
    setSavedRenders,
    appendRunLog,
    setViewMode,
    viewMode,
    layoutVariants,
    variantPickerOpen,
    selectedSavedRenderId,
  } = useEditorStore();
  const evalHooksEnabled = isEvalHooksEnabled();
  const [bootError, setBootError] = useState<string | null>(null);
  const [leftDockOpen, setLeftDockOpen] = useState(true);
  const [rightDockOpen, setRightDockOpen] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);

  useEffect(() => {
    setFloorplan(sampleFloorplan as any);
    setBaseFloorplan(sampleFloorplan as any);
  }, [setBaseFloorplan, setFloorplan]);

  useEffect(() => {
    let cancelled = false;
    const bootstrap = async () => {
      if (evalHooksEnabled) {
        setAuthSession('__tknt_eval__', {
          id: 'eval_user',
          email: 'eval@local',
          display_name: 'Eval User',
          tenant_id: tenantId,
        });
        setAuthResolved(true);
        setInventory([]);
        setSavedLayouts([]);
        setSavedRenders([]);
        setViewMode('3d');
        try {
          const items = await fetchInventory(tenantId);
          if (cancelled) {
            return;
          }
          setInventory(items);
          appendRunLog(`Loaded eval inventory: ${items.length} items`);
        } catch (loadError) {
          if (cancelled) {
            return;
          }
          const message =
            loadError instanceof Error ? loadError.message : 'unknown error';
          setInventory([]);
          appendRunLog(`Eval inventory load failed: ${message}`);
        }
        return;
      }
      try {
        const user = await getCurrentUser();
        if (cancelled) {
          return;
        }
        const token = window.localStorage.getItem('tknt.authToken');
        if (!token) {
          clearStoredAuthToken();
          clearAuthSession();
          return;
        }
        setAuthSession(token, user);
      } catch {
        clearStoredAuthToken();
        clearAuthSession();
      } finally {
        if (!cancelled) {
          setAuthResolved(true);
        }
      }
    };
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [
    clearAuthSession,
    evalHooksEnabled,
    appendRunLog,
    setAuthResolved,
    setAuthSession,
    setInventory,
    setSavedLayouts,
    setSavedRenders,
    setViewMode,
    tenantId,
  ]);

  useEffect(() => {
    if (evalHooksEnabled) {
      return;
    }
    if (!currentUser) {
      return;
    }
    let cancelled = false;
    const loadWorkspace = async () => {
      try {
        const [items, layouts, renders] = await Promise.all([
          fetchInventory(tenantId),
          listSavedLayouts(),
          listGeneratedRenders(),
        ]);
        if (cancelled) {
          return;
        }
        setInventory(items);
        setSavedLayouts(layouts);
        setSavedRenders(renders);
        setBootError(null);
        appendRunLog(`Loaded inventory: ${items.length} items`);
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setInventory([]);
        setSavedLayouts([]);
        setSavedRenders([]);
        const message =
          loadError instanceof Error ? loadError.message : 'unknown error';
        setBootError(message);
        appendRunLog(`Workspace load failed: ${message}`);
      }
    };
    void loadWorkspace();
    return () => {
      cancelled = true;
    };
  }, [
    appendRunLog,
    currentUser,
    setInventory,
    setSavedLayouts,
    setSavedRenders,
    tenantId,
    evalHooksEnabled,
  ]);

  useEffect(() => {
    if (viewMode === '3d' && selectedSavedRenderId) {
      setRightDockOpen(false);
    }
  }, [selectedSavedRenderId, viewMode]);

  useEffect(() => {
    const collapseRightDock = () => setRightDockOpen(false);
    window.addEventListener('tknt:render-focus', collapseRightDock);
    return () => window.removeEventListener('tknt:render-focus', collapseRightDock);
  }, []);

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch {
      // The client-side token is the only thing that matters for ending the session locally.
    } finally {
      clearStoredAuthToken();
      clearAuthSession();
      setBootError(null);
    }
  };

  const workspaceClassName = [
    'workspace',
    viewMode === '3d' ? 'workspace-3d' : 'workspace-2d',
    leftDockOpen ? '' : 'left-dock-collapsed',
    rightDockOpen ? '' : 'right-dock-collapsed',
  ]
    .filter(Boolean)
    .join(' ');

  if (!authResolved) {
    return <div className="app-loading">Loading workspace...</div>;
  }

  if (!currentUser) {
    return <AuthScreen />;
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>TKNT Floorplan Editor</h1>
          <p className="app-subtitle">
            Shared inventory for everyone, plus your own GLB objects, saved layouts,
            and render history.
          </p>
        </div>
        <div className="app-account-actions">
          <button
            className="app-account-chip"
            type="button"
            onClick={() => setProfileOpen(true)}
          >
            <strong>{currentUser.display_name || currentUser.email}</strong>
            <span>{currentUser.email}</span>
          </button>
          <button onClick={handleLogout} type="button">
            Log out
          </button>
        </div>
      </header>
      {profileOpen ? <ProfileModal onClose={() => setProfileOpen(false)} /> : null}
      {bootError ? <div className="app-banner-error">{bootError}</div> : null}
      <Toolbar />
      <div className={workspaceClassName}>
        <div className="left-stack">
          <button
            className="dock-toggle left-dock-toggle"
            type="button"
            onClick={() => setLeftDockOpen((current) => !current)}
            aria-expanded={leftDockOpen}
          >
            <span className="dock-toggle-arrow">{leftDockOpen ? '‹' : '›'}</span>
            <span>{leftDockOpen ? 'Hide' : 'Library'}</span>
          </button>
          <div className="dock-panel-content left-dock-content">
            <LibraryPanel />
          </div>
        </div>
        <div className="canvas-wrapper">
          <div id="three-controls-slot" className="three-controls-slot" />
          <div className="canvas-stage">
            {variantPickerOpen && layoutVariants.length > 0 ? (
              <LayoutVariantsGallery />
            ) : viewMode === '3d' ? (
              <ThreeDPreviewPanel />
            ) : (
              <Canvas2D />
            )}
          </div>
        </div>
        <div className="side-stack">
          <button
            className="dock-toggle right-dock-toggle"
            type="button"
            onClick={() => setRightDockOpen((current) => !current)}
            aria-expanded={rightDockOpen}
          >
            <span className="dock-toggle-arrow">{rightDockOpen ? '›' : '‹'}</span>
            <span>{rightDockOpen ? 'Hide' : 'Workspace'}</span>
          </button>
          <div className="dock-panel-content right-dock-content">
            <UserInputPanel />
            <AccountPanel />
          </div>
        </div>
      </div>
    </div>
  );
};
