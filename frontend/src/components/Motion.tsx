/**
 * Entrance and state-change motion. Everything here is skipped when the system asks for
 * reduced motion (and in environments without matchMedia, such as the test runner).
 */
import { useEffect, useRef, useState, type HTMLAttributes, type ReactNode } from "react";

export function motionAllowed(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: no-preference)").matches;
}

/** rise: fade in from below; unfold: develop downwards; fade: opacity only (the
 * children can key their own animation on the ``.reveal.in`` ancestor). */
type Variant = "rise" | "unfold" | "fade";

/** Plays its entrance once, when it first scrolls into view. The element it renders
 * can itself be a layout item (pass its class name). */
export function Reveal({ children, variant = "rise", delay = 0, className = "", style, as = "div",
                         ...rest }: HTMLAttributes<HTMLElement> & {
  children: ReactNode; variant?: Variant; delay?: number; as?: "div" | "section" | "aside";
}) {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(() => !motionAllowed()
    || typeof IntersectionObserver === "undefined");
  useEffect(() => {
    if (shown || !ref.current) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setShown(true);
        observer.disconnect();
      }
    }, { rootMargin: "0px 0px -8% 0px" });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [shown]);
  const Tag = as;
  return (
    <Tag ref={ref as never} {...rest} className={`reveal ${variant}${shown ? " in" : ""} ${className}`.trim()}
         style={{ ...style, transitionDelay: delay ? `${delay}ms` : undefined }}>
      {children}
    </Tag>
  );
}

/** Number that eases from its previous value (0 on first display) to the new one. */
export function useCountUp(value: number | null, ms = 700): number | null {
  const [shown, setShown] = useState<number | null>(() => (motionAllowed() ? 0 : value));
  const from = useRef(0);
  useEffect(() => {
    if (value === null) return setShown(null);
    if (!motionAllowed()) return setShown(value);
    const start = performance.now();
    const v0 = from.current;
    let frame = 0;
    const step = (now: number) => {
      // The frame timestamp can precede `start`: clamp, or the easing overshoots backwards.
      const t = Math.min(Math.max((now - start) / ms, 0), 1);
      const v = v0 + (value - v0) * (1 - (1 - t) ** 3);
      from.current = v;
      setShown(v);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [value, ms]);
  return shown;
}

/** Integer count-up, or a dash while the value is unknown. */
export function CountUp({ value }: { value: number | null }) {
  const v = useCountUp(value);
  return <>{v === null ? "—" : Math.round(v)}</>;
}
