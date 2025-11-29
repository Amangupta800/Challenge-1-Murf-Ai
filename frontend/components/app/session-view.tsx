'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import type { AppConfig } from '@/app-config';
import { TileLayout } from '@/components/app/tile-layout';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { useChatMessages } from '@/hooks/useChatMessages';
import { useConnectionTimeout } from '@/hooks/useConnectionTimout';
import { useDebugMode } from '@/hooks/useDebug';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/livekit/scroll-area/scroll-area';

const MotionBottom = motion.div;

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut' as const,
  },
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'pointer-events-none h-4 from-background to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  useConnectionTimeout(200_000);
  useDebugMode({ enabled: IN_DEVELOPMENT });

  const messages = useChatMessages();
  const [chatOpen, setChatOpen] = useState(true); // keep the story panel open by default
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: appConfig.supportsChatInput,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsVideoInput,
  };

  // auto-scroll to the latest message
  useEffect(() => {
    if (!scrollAreaRef.current) return;
    scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
  }, [messages.length]);

  const handleRestart = () => {
    // simplest way for Day 8: start a fresh adventure
    window.location.reload();
  };

  return (
    <section
      className="relative z-10 flex h-full w-full items-center justify-center bg-black"
      {...props}
    >
      {/* Centered GM panel */}
      <div className="relative mx-auto flex h-full w-full max-w-5xl flex-col px-4 py-8 md:px-10">
        {/* Top: avatar / bars */}
        <div className="flex items-center justify-center pb-6">
          <TileLayout chatOpen={chatOpen} />
        </div>

        {/* Middle: story card */}
        <div className="relative mx-auto flex w-full max-w-3xl flex-1 flex-col rounded-3xl bg-neutral-950/90 p-6 shadow-[0_0_60px_rgba(0,0,0,0.8)]">
          {/* Header row */}
          <div className="mb-4 flex items-center justify-between gap-4">
            <div className="space-y-1">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-emerald-400">
                Day 8 • Voice Game Master
              </p>
              <h1 className="text-xl font-semibold text-zinc-50">
                The Lost Groves of Eldoria
              </h1>
              <p className="text-xs text-zinc-400">
                Speak your actions. The Game Master continues the story.
              </p>
            </div>

            <button
              type="button"
              onClick={handleRestart}
              className="rounded-full bg-emerald-500 px-4 py-2 text-xs font-semibold text-black shadow-lg transition hover:bg-emerald-400"
            >
              Restart story
            </button>
          </div>

          {/* Transcript area – GM + player speech */}
          <div className="relative mt-2 flex-1 overflow-hidden rounded-2xl bg-zinc-950/80">
            <Fade top className="absolute inset-x-0 top-0 h-8" />
            <ScrollArea
              ref={scrollAreaRef}
              className="max-h-[340px] px-4 pt-6 pb-8"
            >
              <div className="space-y-3">
                {messages.map((m) => {
  const isUser = m.from?.isLocal === true;
  const label = isUser ? 'You' : 'Game Master';

  return (
    <div
      key={m.timestamp}
      className={cn(
        'flex w-full',
        isUser ? 'justify-end' : 'justify-start',
      )}
    >
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
          isUser ? 'bg-emerald-500 text-black' : 'bg-zinc-900 text-zinc-50',
        )}
      >
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.12em] opacity-70">
          {label}
        </div>

        <p className="whitespace-pre-line">
          {m.message}
        </p>
      </div>
    </div>
  );
})}

                          

                {messages.length === 0 && (
                  <p className="text-center text-xs text-zinc-500">
                    When the call starts, your actions and the Game Master&apos;s
                    narration will appear here.
                  </p>
                )}
              </div>
            </ScrollArea>
            <Fade bottom className="pointer-events-none absolute inset-x-0 bottom-0 h-10" />
          </div>
        </div>

        {/* Bottom control bar */}
        <MotionBottom
          {...BOTTOM_VIEW_MOTION_PROPS}
          className="pointer-events-auto fixed inset-x-3 bottom-0 z-50 md:inset-x-12"
        >
          <div className="relative mx-auto max-w-2xl pb-4 md:pb-10">
            <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />
            <AgentControlBar controls={controls} onChatOpenChange={setChatOpen} />
          </div>
        </MotionBottom>
      </div>
    </section>
  );
};
