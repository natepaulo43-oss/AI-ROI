export default function LoadingState() {
  return (
    <div className="bg-gradient-to-br from-[#4a3f35] to-[#3d342a] rounded-[2rem] aspect-square flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-[#b8a894] border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
        <p className="text-[0.7rem] uppercase tracking-[0.15em] text-[#b8a894]">
          Calculating
        </p>
      </div>
    </div>
  );
}
