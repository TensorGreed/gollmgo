package backend

// DeviceInfo describes the GPU device.
type DeviceInfo struct {
	Name             string
	ComputeMajor     int
	ComputeMinor     int
	TotalMemoryBytes uint64
	FreeMemoryBytes  uint64
}
