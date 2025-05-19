<template>
  <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 md:gap-6 xl:grid-cols-4">
    <!-- Report Time Metric Start -->
    <div
        class="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03] md:p-6"
    >
      <div
          class="mb-6 flex h-[52px] w-[52px] items-center justify-center rounded-xl bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-white/[0.90]"
      >
        <svg
            height="24"
            width="24"
            fill="none"
            viewBox="0 0 1024 1024"
            xmlns="http://www.w3.org/2000/svg"
        >
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M971.733333 417.493333a463.573333 463.573333 0 0 0-27.306666-88.106666 450.56 450.56 0 0 0-42.666667-79.786667 42.666667 42.666667 0 1 0-70.613333 47.786667 400.213333 400.213333 0 0 1 35.413333 64 386.133333 386.133333 0 0 1 21.333333 72.106666 386.346667 386.346667 0 0 1-21.333333 226.773334 393.6 393.6 0 0 1-35.413333 65.28 384 384 0 0 1-103.893334 103.893333 382.72 382.72 0 0 1-429.226666 0A386.773333 386.773333 0 0 1 158.08 661.333333a386.133333 386.133333 0 0 1-21.333333-72.106666 386.346667 386.346667 0 0 1 21.333333-226.56 393.6 393.6 0 0 1 35.413333-65.28 384 384 0 0 1 103.893334-103.893334 382.72 382.72 0 0 1 429.226666 0 42.666667 42.666667 0 1 0 47.786667-70.613333 458.24 458.24 0 0 0-79.786667-42.666667 463.573333 463.573333 0 0 0-88.106666-27.306666 469.333333 469.333333 0 0 0-189.013334 0 463.573333 463.573333 0 0 0-88.106666 27.306666 458.24 458.24 0 0 0-79.786667 42.666667 463.146667 463.146667 0 0 0-126.72 126.72 450.56 450.56 0 0 0-42.666667 79.786667 469.333333 469.333333 0 0 0 0 365.226666 450.56 450.56 0 0 0 42.666667 79.786667 463.146667 463.146667 0 0 0 126.72 126.72 458.24 458.24 0 0 0 79.786667 42.666667 469.333333 469.333333 0 0 0 365.226666 0 458.24 458.24 0 0 0 79.786667-42.666667 463.146667 463.146667 0 0 0 126.72-126.72 450.56 450.56 0 0 0 42.666667-79.786667 469.333333 469.333333 0 0 0 27.306666-277.333333z"
              fill="currentColor"/>
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M512 512V256a42.666667 42.666667 0 0 0-85.333333 0v298.666667a42.666667 42.666667 0 0 0 42.666666 42.666666h298.666667a42.666667 42.666667 0 0 0 0-85.333333z"
              fill="currentColor"/>
        </svg>
      </div>

      <p class="text-gray-500 text-theme-sm dark:text-gray-400">上报时间</p>

      <div class="flex items-end justify-between mt-3">
        <div>
          <h4 class="font-bold text-gray-800 text-title-sm dark:text-white/90">{{ time }}</h4>
        </div>
      </div>
    </div>
    <!-- Report Time Metric End -->
    <!-- Report Temperature Metric Start -->
    <div
        class="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03] md:p-6"
    >
      <div
          class="mb-6 flex h-[52px] w-[52px] items-center justify-center rounded-xl bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-white/[0.90]"
      >
        <svg
            width="24"
            height="24"
            fill="none"
            viewBox="0 0 1024 1024"
            xmlns="http://www.w3.org/2000/svg">
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M477.269207 0h69.420975v147.519573h-69.420975zM0 477.269207h147.519573v69.420975H0zM118.501605 851.942889l104.313694-104.305016 49.089307 49.08063-104.305016 104.313693zM118.510283 163.260779l49.089307-49.08063 104.313693 104.305016-49.089307 49.089307zM462.343697 733.692935A234.295792 234.295792 0 0 1 538.01256 277.683902a229.349548 229.349548 0 0 1 34.710488 2.603287V199.585305c0-5.206573 0-10.32637 0.69421-15.446167A340.423109 340.423109 0 0 0 538.01256 182.230061a329.749634 329.749634 0 0 0-72.024262 651.515855 311.526628 311.526628 0 0 1-3.644601-100.052981zM919.827925 568.471013V147.519573a147.519573 147.519573 0 0 0-295.039145 0v420.95144a251.651036 251.651036 0 1 0 295.039145 0zM772.308353 954.538413a182.230061 182.230061 0 0 1-106.73475-329.749633l28.636152-20.826293V147.519573a78.098597 78.098597 0 0 1 156.197195 0v456.356138l28.636152 20.826292A182.230061 182.230061 0 0 1 772.308353 954.538413z"
              fill="currentColor"
          />
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M807.01884 664.966169V338.427256h-69.420975v326.538913a112.809085 112.809085 0 1 0 69.420975 0z"
              fill="currentColor"
          />
        </svg>
      </div>

      <p class="text-gray-500 text-theme-sm dark:text-gray-400">上报气温</p>

      <div class="flex items-end justify-between mt-3">
        <div>
          <h4 class="font-bold text-gray-800 text-title-sm dark:text-white/90">{{ temperature }}℃</h4>
        </div>
      </div>
    </div>
    <!-- Report Temperature Metric End -->

    <!-- Report Humidity Metric Start -->
    <div
        class="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03] md:p-6"
    >
      <div
          class="mb-6 flex h-[52px] w-[52px] items-center justify-center rounded-xl bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-white/[0.90]"
      >
        <svg
            height="24"
            width="24"
            fill="none"
            viewBox="0 0 1024 1024"
            xmlns="http://www.w3.org/2000/svg"
        >
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M659.499267 568.471013V147.519573a147.519573 147.519573 0 0 0-295.039146 0v420.95144a251.651036 251.651036 0 1 0 295.039146 0zM511.979694 954.538413a182.230061 182.230061 0 0 1-106.734749-329.749633l28.636152-20.826293V147.519573a78.098597 78.098597 0 0 1 156.197195 0v456.356138L618.714444 624.78878A182.230061 182.230061 0 0 1 511.979694 954.538413z"
              fill="currentColor"
          />
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M546.690182 664.966169V329.749634h-69.420975v335.216535a112.809085 112.809085 0 1 0 69.420975 0zM978.662202 380.860827c-76.970507-90.420821-84.172933-123.222232-84.867142-127.647819-0.607434 4.512363-7.80986 37.226998-84.780367 127.647819-89.98694 105.693435-37.921208 237.853617 84.172933 243.927953h1.301643c122.007364-6.161112 174.073096-138.234517 84.172933-243.927953z m-69.420975 197.242347a15.272615 15.272615 0 0 1-14.92551 11.975118 16.227153 16.227153 0 0 1-3.471049 0 110.63968 110.63968 0 0 1-87.817534-118.883421 15.185838 15.185838 0 1 1 30.284901 2.776839c0 3.037168-4.946245 70.809395 64.301178 86.77622a15.185838 15.185838 0 0 1 11.541237 17.355244zM250.522945 113.937176C204.27122 59.702039 200.019186 40.003837 199.585305 37.400551c0 2.690063-4.685916 22.301488-50.850865 76.536625C94.759632 177.370592 125.912294 256.684057 199.585305 260.328658h0.780986c72.805248-3.731377 104.044687-82.958066 50.156654-146.391482zM86.776219 401.860672c-0.433881 2.950391-5.206573 24.817999-56.491318 85.040695C-29.764243 557.363657 4.946245 645.528296 86.776219 649.520002h0.867763c81.396094-4.078482 116.106582-92.156345 56.144214-162.618635-51.805403-60.222696-56.578095-82.090304-57.011977-85.040695z"
              fill="currentColor"
          />
        </svg>
      </div>

      <p class="text-gray-500 text-theme-sm dark:text-gray-400">上报湿度</p>

      <div class="flex items-end justify-between mt-3">
        <div>
          <h4 class="font-bold text-gray-800 text-title-sm dark:text-white/90">{{ humidity }}%</h4>
        </div>
      </div>
    </div>
    <!-- Report Humidity Metric End -->

    <!-- Report Windpower Metric Start -->
    <div
        class="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03] md:p-6"
    >
      <div
          class="mb-6 flex h-[52px] w-[52px] items-center justify-center rounded-xl bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-white/[0.90]"
      >
        <svg
            height="24"
            width="24"
            viewBox="0 0 1024 1024"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
          <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M746.305085 208.271186H538.033898v277.694916h485.966102a277.694915 277.694915 0 0 0-277.694915-277.694916zM633.491525 390.508475V303.728814h112.81356a182.237288 182.237288 0 0 1 155.162034 86.779661zM208.271186 277.694915v208.271187h277.694916V0a277.694915 277.694915 0 0 0-277.694916 277.694915z m95.457628 0a182.237288 182.237288 0 0 1 86.779661-155.162034V390.508475H303.728814zM277.694915 815.728814h208.271187V538.033898H0a277.694915 277.694915 0 0 0 277.694915 277.694916z m112.81356-182.237289v86.779661H277.694915a182.237288 182.237288 0 0 1-155.162034-86.779661zM538.033898 1024a277.694915 277.694915 0 0 0 277.694916-277.694915V538.033898H538.033898z m95.457627-390.508475h86.779661v112.81356a182.237288 182.237288 0 0 1-86.779661 155.162034z"
              fill="currentColor"/>
        </svg>
      </div>

      <p class="text-gray-500 text-theme-sm dark:text-gray-400">上报风速</p>

      <div class="flex items-end justify-between mt-3">
        <div>
          <h4 class="font-bold text-gray-800 text-title-sm dark:text-white/90">{{ windpower }}m/s</h4>
        </div>
      </div>
    </div>
    <!-- Metric Item End -->
  </div>
</template>
<script setup lang="js">
defineProps({
  time: {
    type: String,
    required: true
  },
  temperature: {
    type: Number,
    required: true
  },
  humidity: {
    type: Number,
    required: true
  },
  windpower: {
    type: Number,
    required: true
  }
})
</script>