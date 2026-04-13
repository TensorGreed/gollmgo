package scheduler

// Policy selects the scheduling algorithm.
type Policy int

const (
	PolicyFCFS     Policy = iota // First-come, first-served
	PolicySJF                    // Shortest job first
	PolicyPriority               // Priority-based (highest first)
)

func (p Policy) String() string {
	switch p {
	case PolicyFCFS:
		return "FCFS"
	case PolicySJF:
		return "SJF"
	case PolicyPriority:
		return "PRIORITY"
	default:
		return "UNKNOWN"
	}
}

// SchedulerConfig is the unified configuration for all scheduler policies.
type SchedulerConfig struct {
	MaxBatchSize     int // max sequences per batch
	MaxTokenBudget   int // max total tokens (prefill + decode) per tick
	MaxQueueDepth    int // max waiting queue length (0 = unbounded)
	PrefillChunkSize int // max prefill tokens per sequence per tick (0 = no chunking)

	// Priority-specific options.
	AutoPreempt bool // if true, high-priority arrivals can preempt low-priority active sequences

	// Preemption mode.
	PreemptMode PreemptMode

	// SJF starvation prevention: after MaxWaitTicks ticks in the waiting queue,
	// a sequence is promoted to the front regardless of job size. 0 = disabled.
	MaxWaitTicks int
}

// DefaultSchedulerConfig returns reasonable defaults.
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		MaxBatchSize:     64,
		MaxTokenBudget:   4096,
		MaxQueueDepth:    256,
		PrefillChunkSize: 512,
		PreemptMode:      PreemptRecompute,
		MaxWaitTicks:     100,
	}
}

// NewScheduler creates a scheduler for the given policy.
func NewScheduler(policy Policy, cfg SchedulerConfig) Scheduler {
	switch policy {
	case PolicySJF:
		return NewSJFScheduler(cfg)
	case PolicyPriority:
		return NewPriorityScheduler(cfg)
	default:
		return NewFCFSScheduler(FCFSConfig{
			MaxBatchSize:     cfg.MaxBatchSize,
			MaxTokenBudget:   cfg.MaxTokenBudget,
			MaxQueueDepth:    cfg.MaxQueueDepth,
			PrefillChunkSize: cfg.PrefillChunkSize,
		})
	}
}
