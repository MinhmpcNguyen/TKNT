import React from 'react';

export const Layout: React.FC<React.PropsWithChildren> = ({ children }) => {
  return <div className="app-shell">{children}</div>;
};
